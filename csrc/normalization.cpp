/*
 * normalization.cpp — Custom Layer Normalization Kernel
 *
 * WHY LAYER NORMALIZATION?
 * -------------------------
 * Deep networks suffer from "internal covariate shift": the distribution of
 * each layer's inputs changes as weights update, making training unstable.
 * LayerNorm fixes this by normalizing WITHIN each sample across its features,
 * ensuring each sample always has mean≈0, std≈1 before the affine step.
 *
 * Unlike BatchNorm (which normalizes across the batch), LayerNorm normalizes
 * across features for each sample independently. This makes it:
 *   ✓ Batch-size independent (works with batch=1)
 *   ✓ Better for sequences where different positions have different statistics
 *   ✓ The standard choice for Transformers (GPT, BERT, etc.)
 *
 * MATH:
 *   Given input x ∈ ℝ^C (one row of the batch):
 *
 *     μ   = (1/C) * Σ x_c              mean
 *     σ²  = (1/C) * Σ (x_c - μ)²      variance
 *     x̂_c = (x_c - μ) / √(σ² + ε)    normalized (zero mean, unit variance)
 *     y_c = γ_c * x̂_c + β_c           affine re-scale and shift
 *
 *   ε (epsilon) is a small constant (1e-5) for numerical stability —
 *   prevents division by zero when variance is near zero.
 *   γ (weight) and β (bias) are LEARNED parameters, giving the network
 *   the ability to undo normalization if optimal.
 *
 * PASS COUNT HISTORY AND WHY IT MATTERS:
 *   Naive: 4 passes (mean, var, normalize, affine)
 *   v1:    3 passes (mean, var, fused normalize+affine)
 *   v2:    2 passes (fused mean+var, fused normalize+affine) ← this version
 *
 *   Each pass over C=2048 floats reads 8KB from cache. Reducing from 3→2 passes
 *   saves one full scan per row. For N=256, C=1024: saves 1MB of reads, ~20%
 *   bandwidth reduction → ~20% faster forward pass.
 *
 * TRICK: E[X²] - E[X]² for single-pass variance
 *   Var(X) = E[X²] - E[X]² = (Σx²/C) - (Σx/C)²
 *   By accumulating sum and sum-of-squares simultaneously in one loop, we get
 *   both mean and variance from a single pass.
 *
 *   Why is this safe when Welford (also single-pass) is not?
 *   Welford has a loop-carried dependency: mean[c] depends on mean[c-1].
 *   This breaks SIMD. Our two-accumulator approach:
 *     sum    += x        ← no cross-iteration dependency
 *     sq_sum += x * x    ← no cross-iteration dependency
 *   Both reductions are independent and SIMD-vectorizable.
 *
 *   Numerical note: E[X²] - E[X]² can lose precision when mean² >> variance
 *   (large mean, tiny variance). For neural network activations (values in
 *   [-6, 6], mean ≈ 0), this is not a concern in practice. We clamp the
 *   result to ≥0 to handle any floating-point rounding to a tiny negative.
 *
 * BACKWARD DERIVATION:
 *   The backward pass is non-trivial because μ and σ² both depend on ALL
 *   elements of x, so the gradient from every output y_c flows back to
 *   every input x_c' through the shared mean and variance.
 *
 *   Let x̂_c = (x_c - μ) * rstd,  y_c = γ_c * x̂_c + β_c
 *   Given upstream gradient dy_c (grad_out):
 *
 *   dL/dγ_c = Σ_i  dy_c^(i) * x̂_c^(i)       (sum over batch samples i)
 *   dL/dβ_c = Σ_i  dy_c^(i)
 *
 *   For grad_input, defining:
 *     sum1 = Σ_c (dy_c * γ_c)
 *     sum2 = Σ_c (dy_c * γ_c * (x_c - μ))
 *
 *   The final gradient per input element:
 *     dL/dx_c = rstd * [ dy_c*γ_c - (1/C)*sum1 - (1/C)*(x_c-μ)*rstd²*sum2 ]
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <algorithm>  // std::max


// ─── FORWARD PASS ─────────────────────────────────────────────────────────────
/*
 * Returns a tuple of three tensors:
 *   output [N, C]  — normalized + affine output
 *   mean   [N]     — per-sample mean, saved for backward
 *   rstd   [N]     — per-sample reciprocal std, saved for backward
 *
 * Saving mean and rstd avoids recomputing them in backward (just 2*N floats).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_forward(
    const torch::Tensor& input,   // [N, C]  — batch of N feature vectors
    const torch::Tensor& weight,  // [C]     — learnable γ (scale)
    const torch::Tensor& bias,    // [C]     — learnable β (shift)
    float eps)                    // numerical stability constant (default 1e-5)
{
    // ── Input validation ──────────────────────────────────────────────────────
    TORCH_CHECK(input.dim() == 2,   "input must be 2D [batch, features]");
    TORCH_CHECK(weight.dim() == 1,  "weight must be 1D");
    TORCH_CHECK(bias.dim() == 1,    "bias must be 1D");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(),   "bias must be contiguous");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.scalar_type()   == torch::kFloat32, "bias must be float32");

    const int64_t N = input.size(0);  // batch size
    const int64_t C = input.size(1);  // number of features (channels)
    TORCH_CHECK(weight.size(0) == C, "weight size must match features");
    TORCH_CHECK(bias.size(0)   == C, "bias size must match features");

    auto out  = torch::empty_like(input);           // [N, C] output
    auto mean = torch::empty({N}, input.options()); // [N] mean per sample
    auto rstd = torch::empty({N}, input.options()); // [N] 1/std per sample

    const float* __restrict__ inp = input.data_ptr<float>();
    const float* __restrict__ w   = weight.data_ptr<float>();
    const float* __restrict__ b   = bias.data_ptr<float>();
    float*       __restrict__ o   = out.data_ptr<float>();
    float*       __restrict__ m   = mean.data_ptr<float>();
    float*       __restrict__ r   = rstd.data_ptr<float>();

    // Hoist the 1/C multiplication factor outside all loops.
    // Division is ~20-40x more expensive than multiplication on most CPUs.
    // Hoisting it here means we do one division total instead of one per row.
    const float inv_C = 1.0f / static_cast<float>(C);

    // ── Parallel loop: each sample i is fully independent ────────────────────
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float* row = inp + i * C;  // pointer to input sample i
            float*       dst = o   + i * C;  // pointer to output sample i

            // ── Fused Pass 1: mean AND variance in a single loop ──────────────
            //
            // ALGORITHM: accumulate sum=Σx and sq_sum=Σx² simultaneously.
            //   mean = sum / C
            //   var  = sq_sum/C - mean²     (Var(X) = E[X²] - E[X]²)
            //
            // WHY THIS IS VECTORIZABLE (unlike Welford):
            //   sum    += row[c]           ← no dependency between iterations
            //   sq_sum += row[c] * row[c]  ← no dependency between iterations
            //
            // The compiler can issue these as two parallel SIMD reduction chains:
            //   vsum   = vadd(vsum,   vload(row + c))    ; 8 floats at once (NEON/AVX2)
            //   vsqsum = vfma(vsqsum, vload(row+c), vload(row+c))
            //
            // Old v1 had two sequential loops over C (one for mean, one for var).
            // This fused loop reads each element ONCE → saves one full row read.
            // For C=2048 (8KB/row): saves 8KB of cache/memory reads per sample.
            float sum    = 0.0f;
            float sq_sum = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                float x  = row[c];
                sum    += x;
                sq_sum += x * x;
            }

            float mu  = sum * inv_C;

            // Var(X) = E[X²] - E[X]²
            // std::max(0, ...) guards against tiny negative values from
            // floating-point rounding (e.g. var=-1e-8 when all inputs are equal).
            // Without this, std::sqrt(negative) = NaN.
            float var = std::max(0.0f, sq_sum * inv_C - mu * mu);

            // rstd = 1 / sqrt(σ² + ε).  Store reciprocal to use multiply in backward.
            float rs = 1.0f / std::sqrt(var + eps);

            // Save per-sample statistics for the backward pass.
            m[i] = mu;
            r[i] = rs;

            // ── Pass 2: Normalize + affine transform ──────────────────────────
            // Fused: y_c = (x_c - μ) * rstd * γ_c + β_c
            // Reads: row[c], w[c], b[c]. Writes: dst[c].
            // No intermediate x̂ tensor — stays in registers.
            for (int64_t c = 0; c < C; ++c) {
                dst[c] = (row[c] - mu) * rs * w[c] + b[c];
            }
        }
    });

    return {out, mean, rstd};
}


// ─── BACKWARD PASS ───────────────────────────────────────────────────────────
/*
 * Compute gradients w.r.t. input, weight (γ), and bias (β).
 *
 * The tricky part: normalizing couples all C features within a sample.
 * Changing x_c changes the mean and variance, which affects ALL x̂_c'.
 *
 * PARALLELISM STRATEGY:
 *   grad_input:  trivially parallel over N rows (each row independent)
 *   grad_weight: must sum over all N rows — previously sequential
 *   grad_bias:   must sum over all N rows — previously sequential
 *
 * The old sequential grad_weight loop ran N*C iterations on one thread.
 * For N=1024, C=2048: 2M sequential iterations ≈ 0.5ms wasted.
 *
 * Fix: each thread accumulates into its own private partial buffer,
 * then we reduce across threads with a single torch::sum(0) call.
 * Memory cost: num_threads * C * 4 bytes (e.g., 8 * 2048 * 4 = 64KB) —
 * trivial compared to the N*C input tensor.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_backward(
    const torch::Tensor& grad_out,  // [N, C] — dL/dY from upstream
    const torch::Tensor& input,     // [N, C] — saved from forward
    const torch::Tensor& weight,    // [C]    — γ, saved from forward
    const torch::Tensor& mean,      // [N]    — μ per sample, saved from forward
    const torch::Tensor& rstd,      // [N]    — 1/σ per sample, saved from forward
    float /*eps*/)                  // not needed: rstd already incorporates ε
{
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(input.is_contiguous(),    "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),   "weight must be contiguous");

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);

    auto grad_input = torch::empty_like(input);  // [N, C]

    const float* __restrict__ go  = grad_out.data_ptr<float>();
    const float* __restrict__ inp = input.data_ptr<float>();
    const float* __restrict__ w   = weight.data_ptr<float>();
    const float* __restrict__ mu  = mean.data_ptr<float>();
    const float* __restrict__ rs  = rstd.data_ptr<float>();
    float*       __restrict__ gi  = grad_input.data_ptr<float>();

    // ── Parallel grad_weight and grad_bias ────────────────────────────────────
    // Each thread accumulates into a private row of the partial buffer.
    // No atomics needed — threads never share a row.
    // After the parallel loop, sum across the thread dimension.
    //
    // thread_gw[tid, c] = Σ_{i in thread's range}  dy[i,c] * x̂[i,c]
    // thread_gb[tid, c] = Σ_{i in thread's range}  dy[i,c]
    //
    // Final: grad_weight = thread_gw.sum(0),  grad_bias = thread_gb.sum(0)
    const int64_t T = at::get_num_threads();  // number of CPU threads in pool
    auto thread_gw = torch::zeros({T, C}, input.options()); // [T, C]
    auto thread_gb = torch::zeros({T, C}, input.options()); // [T, C]
    float* tgw = thread_gw.data_ptr<float>();
    float* tgb = thread_gb.data_ptr<float>();

    const float inv_C = 1.0f / static_cast<float>(C);

    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        // at::get_thread_num() returns which thread (0 to T-1) is running this lambda.
        // Each thread writes to a different row of tgw/tgb — no false sharing.
        const int64_t tid = at::get_thread_num();
        float* my_gw = tgw + tid * C;  // this thread's private grad_weight buffer
        float* my_gb = tgb + tid * C;  // this thread's private grad_bias buffer

        for (int64_t i = begin; i < end; ++i) {
            const float* row_go  = go  + i * C;
            const float* row_inp = inp + i * C;
            float mi = mu[i], ri = rs[i];

            // Accumulate into thread-private buffers (no synchronization needed).
            // dL/dγ_c = dy[i,c] * x̂[i,c]  where x̂[i,c] = (x[i,c]-μ[i]) * rstd[i]
            // dL/dβ_c = dy[i,c]
            for (int64_t c = 0; c < C; ++c) {
                float x_hat  = (row_inp[c] - mi) * ri;  // normalized input
                my_gw[c]    += row_go[c] * x_hat;        // γ gradient
                my_gb[c]    += row_go[c];                 // β gradient
            }
        }
    });

    // Reduce: sum each thread's partial result across the thread dimension.
    // torch::sum(0) efficiently collapses the T rows into one [C] vector.
    auto grad_weight = thread_gw.sum(0);  // [C]
    auto grad_bias   = thread_gb.sum(0);  // [C]

    // ── Parallel grad_input: each row is independent ──────────────────────────
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float* row_go  = go  + i * C;
            const float* row_inp = inp + i * C;
            float*       row_gi  = gi  + i * C;
            float mi = mu[i], ri = rs[i];

            // Pass 1: compute the two correction sums.
            //
            // sum1 = Σ_c (dy_c * γ_c)             — "mean of weighted grads"
            // sum2 = Σ_c (dy_c * γ_c * (x_c - μ)) — "covariance of grads with x"
            //
            // These account for the fact that nudging any single x_c changes the
            // mean and variance, which then shifts ALL other normalized outputs.
            // Without these correction terms, gradients would be wrong.
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                float dy_w  = row_go[c] * w[c];       // dL/dx̂_c = dy_c * γ_c
                sum1       += dy_w;
                sum2       += dy_w * (row_inp[c] - mi);
            }

            // Pass 2: apply the full LayerNorm backward formula.
            //   dL/dx_c = rstd * [ dL/dx̂_c
            //                     - (1/C)*sum1             ← mean correction
            //                     - (1/C)*(x_c-μ)*rstd²*sum2 ← variance correction ]
            for (int64_t c = 0; c < C; ++c) {
                float x_mu = row_inp[c] - mi;
                float dy_w = row_go[c] * w[c];
                row_gi[c]  = ri * (dy_w
                                   - inv_C * sum1
                                   - inv_C * x_mu * ri * ri * sum2);
            }
        }
    });

    return {grad_input, grad_weight, grad_bias};
}


// ─── Python bindings ─────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &layer_norm_forward,  "LayerNorm forward");
    m.def("backward", &layer_norm_backward, "LayerNorm backward");
}
