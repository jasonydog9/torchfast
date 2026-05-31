/*
 * fused_linear_act.cpp — Fused Linear + Activation Kernel
 *
 * WHY FUSION?
 * -----------
 * A standard PyTorch nn.Linear followed by nn.GELU does this:
 *   1. Compute Z = X @ W.T + b  → write Z to memory (RAM/cache)
 *   2. Read Z back from memory
 *   3. Compute act(Z)           → write result to memory again
 *
 * Modern CPUs (and GPUs) are bottlenecked by MEMORY BANDWIDTH, not arithmetic.
 * Reading/writing large tensors is expensive. Fusion eliminates the intermediate
 * write+read of Z by computing the dot product and activation IN THE SAME LOOP,
 * keeping the partial sums in CPU registers the whole time.
 *
 * In short: fewer memory round-trips = faster execution.
 *
 * MATH OVERVIEW — Linear Layer:
 *   Given input X of shape [N, K] and weight W of shape [M, K]:
 *   output[i, j] = bias[j] + sum_{k=0}^{K-1}  X[i,k] * W[j,k]
 *
 *   This is a matrix multiply (GEMM). Each output neuron j computes a dot
 *   product between the i-th input row and the j-th weight row.
 *
 * MATH OVERVIEW — Activation Functions:
 *   Activations introduce non-linearity. Without them, stacking linear layers
 *   would just be one big linear layer (matrix multiply chains collapse).
 *
 * BACKPROPAGATION OVERVIEW:
 *   We need gradients of the loss L w.r.t. every parameter.
 *   The chain rule says:
 *     dL/dX    = dL/dZ * dZ/dX     (where Z = act(X @ W.T + b))
 *     dL/dW    = dL/dZ * dZ/dW
 *     dL/db    = dL/dZ * dZ/db
 *
 *   dL/dZ is grad_out (comes from the next layer's backward pass).
 *   dZ/d(pre_act) is act'(pre_act), the activation's derivative.
 *
 *   So d_pre[i,j] = grad_out[i,j] * act'(pre_act[i,j])   (elementwise)
 *   Then:
 *     grad_input  = d_pre @ W        shape [N, K]
 *     grad_weight = d_pre.T @ X      shape [M, K]
 *     grad_bias   = d_pre.sum(dim=0) shape [M]
 */

#include <torch/extension.h>   // pulls in all of ATen + pybind11
#include <ATen/Parallel.h>     // at::parallel_for — CPU thread pool
#include <cmath>               // std::tanh, std::exp, std::sqrt

// ─── Compile-time constants ──────────────────────────────────────────────────
// These are exact float32 literals baked at compile time — faster than
// computing them at runtime and avoids magic numbers scattered in the code.

static constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2 / pi)
static constexpr float kGELUCoeff    = 0.044715f;           // empirical fit coeff


// ─── GELU (Gaussian Error Linear Unit) ───────────────────────────────────────
/*
 * GELU is the default activation in GPT, BERT, and most modern transformers.
 * Unlike ReLU (hard gate), GELU is a SMOOTH gate: it weights x by the
 * probability that x is greater than a standard Gaussian.
 *
 * Exact form:  GELU(x) = x * Phi(x)   where Phi = CDF of N(0,1)
 * Phi involves erf(), which is expensive. So we use the tanh approximation:
 *
 *   GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
 *
 * The 0.044715 coefficient was found empirically to make the approximation
 * tight across the range of typical pre-activation values.
 *
 * Visual intuition:
 *   x < -3:  ≈ 0      (neuron off)
 *   x ≈ 0:   ≈ 0      (near-off, smooth)
 *   x > 3:   ≈ x      (neuron fully on, linear passthrough)
 *   The transition around 0 is soft / differentiable everywhere.
 */
inline float gelu_fwd(float x) {
    // inner = sqrt(2/π) * (x + 0.044715*x³)
    // This polynomial closely approximates the argument to tanh in exact GELU.
    float inner = kSqrt2OverPi * (x + kGELUCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

/*
 * GELU backward — derivative via chain rule.
 *
 * Let inner = sqrt(2/π) * (x + c*x³)    where c = 0.044715
 * Let t = tanh(inner)
 *
 * GELU(x) = 0.5 * x * (1 + t)
 *
 * d/dx [0.5 * x * (1 + t)]
 *   = 0.5*(1 + t) + 0.5*x * dt/dx        (product rule)
 *
 * dt/dx = sech²(inner) * d(inner)/dx     (chain rule through tanh)
 *   sech²(inner) = 1 - tanh²(inner) = 1 - t²
 *   d(inner)/dx  = sqrt(2/π) * (1 + 3*c*x²)
 *
 * Putting it together:
 *   GELU'(x) = 0.5*(1+t) + 0.5*x*(1-t²)*sqrt(2/π)*(1+3*c*x²)
 *
 * We pass this back to be multiplied by grad_out in the backward pass.
 */
inline float gelu_bwd(float x) {
    float inner  = kSqrt2OverPi * (x + kGELUCoeff * x * x * x);
    float tanh_v = std::tanh(inner);
    float sech2  = 1.0f - tanh_v * tanh_v;  // sech²(inner) = 1 - tanh²(inner)
    float dtanh  = kSqrt2OverPi * (1.0f + 3.0f * kGELUCoeff * x * x); // d(inner)/dx
    return 0.5f * (1.0f + tanh_v) + 0.5f * x * sech2 * dtanh;
}


// ─── ReLU (Rectified Linear Unit) ────────────────────────────────────────────
/*
 * The simplest activation: pass positive values, zero out negatives.
 * ReLU(x) = max(0, x)
 *
 * Advantages: cheap, no saturation for positive inputs, sparse activations.
 * Disadvantages: "dying ReLU" problem — neurons stuck at 0 can't recover.
 *
 * Derivative: 1 if x > 0, else 0  (not differentiable at x=0, but we pick 0).
 * This is the "subgradient" commonly used in practice.
 */
inline float relu_fwd(float x) { return x > 0.0f ? x : 0.0f; }
inline float relu_bwd(float x) { return x > 0.0f ? 1.0f : 0.0f; }


// ─── SiLU / Swish ────────────────────────────────────────────────────────────
/*
 * SiLU (Sigmoid Linear Unit), also called Swish.
 * Used in LLaMA, Gemma, EfficientNet, etc.
 *
 * SiLU(x) = x * σ(x)   where σ(x) = 1/(1+e^{-x})  (sigmoid)
 *
 * Intuition: sigmoid acts as a soft gate on x (similar to GELU but simpler
 * to compute). The neuron "gates itself" — near 0 for very negative inputs,
 * near x for very positive inputs.
 *
 * SiLU'(x) = σ(x) + x*σ(x)*(1 - σ(x))
 *           = σ(x) * (1 + x*(1 - σ(x)))
 *           = σ(x) * (1 + x - x*σ(x))
 *
 * We compute σ(x) once and reuse it for efficiency.
 */
inline float silu_fwd(float x) {
    return x / (1.0f + std::exp(-x));  // x * sigmoid(x), rewritten to avoid overflow
}
inline float silu_bwd(float x) {
    float sig = 1.0f / (1.0f + std::exp(-x));  // σ(x)
    return sig * (1.0f + x * (1.0f - sig));     // SiLU'(x)
}


// ─── FORWARD PASS ─────────────────────────────────────────────────────────────
/*
 * Fused computation: output[i,j] = act( dot(input[i,:], weight[j,:]) + bias[j] )
 *
 * Standard approach (unfused):
 *   pre = input @ weight.T + bias   ← writes [N, M] tensor to RAM
 *   out = act(pre)                   ← reads [N, M] tensor back from RAM
 *
 * Fused approach (this function):
 *   For each (i, j):
 *     acc = bias[j]
 *     for k in range(K): acc += input[i,k] * weight[j,k]
 *     out[i,j] = act(acc)            ← acc lives in a CPU register the whole time
 *
 * The intermediate [N, M] tensor is never materialized.
 * Memory bandwidth saved: 2 * N * M * 4 bytes (one write + one read).
 * For (1024, 2048) shapes: that's ~16 MB saved per forward pass.
 */
torch::Tensor fused_linear_act_forward(
    const torch::Tensor& input,   // [N, in_features]  — batch of N samples
    const torch::Tensor& weight,  // [out_features, in_features]  — learned W
    const torch::Tensor& bias,    // [out_features]  — learned b
    const std::string&   act)     // which activation: "gelu", "relu", "silu"
{
    // ── Input validation ──────────────────────────────────────────────────────
    // TORCH_CHECK is the PyTorch way to validate — throws a Python-visible
    // RuntimeError with a helpful message instead of crashing with a segfault.
    TORCH_CHECK(input.dim() == 2,  "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1,   "bias must be 1D");

    // Contiguous = elements are laid out sequentially in memory (C row-major order).
    // If a tensor was transposed or sliced, its memory layout may not match its
    // logical shape. data_ptr<float>() only makes sense on contiguous tensors.
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(),   "bias must be contiguous");

    // Our kernel uses float* arithmetic — we don't support mixed precision here.
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.scalar_type()   == torch::kFloat32, "bias must be float32");

    const int64_t N = input.size(0);   // batch size
    const int64_t K = input.size(1);   // input features
    const int64_t M = weight.size(0);  // output features

    TORCH_CHECK(weight.size(1) == K, "weight inner dim must match input features");
    TORCH_CHECK(bias.size(0)   == M, "bias size must match out_features");
    TORCH_CHECK(act == "gelu" || act == "relu" || act == "silu",
                "act must be 'gelu', 'relu', or 'silu'");

    // Allocate output tensor with same device/dtype as input (no initialization)
    auto out = torch::empty({N, M}, input.options());

    // ── Raw pointer extraction ────────────────────────────────────────────────
    // data_ptr<float>() gives us the underlying C float* pointer.
    // __restrict__ is a compiler hint: these pointers do NOT alias each other.
    // This is critical for auto-vectorization (SIMD) — without it, the compiler
    // must assume any write to out_ptr could affect inp_ptr and can't reorder ops.
    const float* __restrict__ inp_ptr = input.data_ptr<float>();
    const float* __restrict__ wgt_ptr = weight.data_ptr<float>();
    const float* __restrict__ bia_ptr = bias.data_ptr<float>();
    float*       __restrict__ out_ptr = out.data_ptr<float>();

    // ── Parallel loop over batch dimension ───────────────────────────────────
    // at::parallel_for splits [0, N) across available CPU threads (e.g. 8 threads
    // on an 8-core machine each handle N/8 rows).
    // The grain_size=0 means "let PyTorch decide the optimal chunk size".
    // Each thread processes a contiguous range [begin, end) of rows — no false sharing.
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            // Pointer to the start of input row i.
            // Memory layout: input[i,k] = inp_ptr[i*K + k]
            const float* row = inp_ptr + i * K;

            // Pointer to the start of output row i.
            float*       o   = out_ptr + i * M;

            // For each output neuron j, compute the fused linear+activation
            for (int64_t j = 0; j < M; ++j) {
                // Pointer to weight row j (the j-th output neuron's weights).
                // weight[j,k] = wgt_ptr[j*K + k]
                const float* w = wgt_ptr + j * K;

                // Start accumulator at bias[j] so we only need one pass
                float acc = bia_ptr[j];

                // Dot product: acc = bias[j] + sum_k input[i,k] * weight[j,k]
                // This inner loop is the hot path — the compiler will auto-vectorize
                // with SIMD (process 8 floats at once with AVX2) thanks to __restrict__.
                for (int64_t k = 0; k < K; ++k) acc += row[k] * w[k];

                // Apply activation IN PLACE — acc never touches memory as Z.
                // This is the "fusion": activation applied before writing to output.
                if      (act == "gelu") o[j] = gelu_fwd(acc);
                else if (act == "relu") o[j] = relu_fwd(acc);
                else                   o[j] = silu_fwd(acc);
            }
        }
    });
    return out;
}


// ─── BACKWARD PASS ───────────────────────────────────────────────────────────
/*
 * Backpropagation for the fused Linear + Activation layer.
 *
 * Notation:
 *   Z    = X @ W.T + b       (pre-activation, shape [N, M])
 *   Y    = act(Z)            (output, shape [N, M])
 *   L    = loss scalar
 *   dL   = grad_out          (dL/dY from upstream, shape [N, M])
 *
 * Step 1 — Backprop through activation (elementwise):
 *   dL/dZ[i,j] = dL/dY[i,j] * act'(Z[i,j])
 *   Call this d_pre ("gradient of pre-activation").
 *
 * Step 2 — Backprop through the linear layer:
 *   dL/dX    = d_pre @ W          [N,M] @ [M,K] = [N,K]
 *   dL/dW    = d_pre.T @ X        [M,N] @ [N,K] = [M,K]
 *   dL/db    = d_pre.sum(axis=0)  sum over batch = [M]
 *
 * Why recompute Z here instead of saving it in forward?
 *   Saving Z costs N*M*4 bytes of memory (e.g. 8 MB for (1024, 2048)).
 *   Recomputing costs some FLOPs but saves memory. This is the "checkpointing"
 *   tradeoff. For CPU inference the recompute cost is cheap.
 *
 * Note: We use PyTorch's torch::mm / torch::addmm for the matrix multiplies in
 * the backward pass (correctness > micro-optimization in backward).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_linear_act_backward(
    const torch::Tensor& grad_out,  // [N, M] — dL/dY from next layer
    const torch::Tensor& input,     // [N, K] — saved from forward
    const torch::Tensor& weight,    // [M, K] — saved from forward
    const torch::Tensor& bias,      // [M]    — saved from forward
    const std::string&   act)
{
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(input.is_contiguous(),    "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),   "weight must be contiguous");

    const int64_t N = input.size(0);
    const int64_t K = input.size(1);
    const int64_t M = weight.size(0);

    // ── Step 1a: Recompute pre-activation Z = X @ W.T + b ────────────────────
    // torch::addmm(bias, A, B) computes  bias + A @ B  in one fused BLAS call.
    // weight.t() transposes [M, K] → [K, M], so input[N,K] @ weight.t()[K,M] = [N,M].
    auto pre_act = torch::addmm(bias, input, weight.t()); // [N, M]

    // ── Step 1b: Elementwise multiply grad_out by activation derivative ───────
    // d_pre[i,j] = grad_out[i,j] * act'(Z[i,j])
    // This is the chain rule passing the gradient "through" the activation gate.
    auto d_pre = torch::empty_like(pre_act);

    const float* __restrict__ go  = grad_out.data_ptr<float>();
    const float* __restrict__ pre = pre_act.data_ptr<float>();
    float*       __restrict__ dp  = d_pre.data_ptr<float>();

    // Parallelize over ALL N*M elements (row-major, fully flat).
    // Each element is independent — perfect data parallelism.
    at::parallel_for(0, N * M, 0, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            float da;  // activation derivative at this element
            if      (act == "gelu") da = gelu_bwd(pre[idx]);
            else if (act == "relu") da = relu_bwd(pre[idx]);
            else                    da = silu_bwd(pre[idx]);
            dp[idx] = go[idx] * da;  // chain rule: dL/dZ = dL/dY * dY/dZ
        }
    });

    // ── Step 2: Backprop through the linear transform ─────────────────────────
    // torch::mm delegates to BLAS (OpenBLAS/MKL) which is highly optimized.

    // grad_input[N,K] = d_pre[N,M] @ weight[M,K]
    // Each input element[i,k] receives gradient contributions from ALL M output neurons.
    auto grad_input  = torch::mm(d_pre, weight);

    // grad_weight[M,K] = d_pre.T[M,N] @ input[N,K]
    // Each weight w[j,k] contributed to every sample's output — sum over batch.
    auto grad_weight = torch::mm(d_pre.t(), input);

    // grad_bias[M] = sum over all N rows of d_pre
    // Bias is broadcast across the batch, so gradient accumulates from all N rows.
    auto grad_bias   = d_pre.sum(0);

    return {grad_input, grad_weight, grad_bias};
}


// ─── Python bindings ─────────────────────────────────────────────────────────
// PYBIND11_MODULE registers this C++ shared library as a Python module.
// TORCH_EXTENSION_NAME is replaced by the name given in setup.py ("fused_linear_act").
// After `pip install .`, Python can do:  import fused_linear_act; fused_linear_act.forward(...)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &fused_linear_act_forward,  "Fused Linear+Activation forward");
    m.def("backward", &fused_linear_act_backward, "Fused Linear+Activation backward");
}
