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
 * WHY FUSION HELPS HERE:
 *   Unfused approach: 4 passes over the data
 *     Pass 1: compute mean   (read x)
 *     Pass 2: compute var    (read x again)
 *     Pass 3: normalize      (read x, mean, var; write x_hat)
 *     Pass 4: affine         (read x_hat; write y)
 *
 *   Our fused approach: 3 passes (mean, variance, normalize+affine)
 *   — still reads x twice (once for mean, once for var+normalize+affine)
 *   but avoids materializing the intermediate x_hat tensor.
 *
 * PERFORMANCE TRICK — Store rstd, not std:
 *   We save  rstd = 1 / √(σ² + ε)  instead of σ² in the forward pass.
 *   The backward pass needs to divide by σ anyway, and recomputing the
 *   sqrt + division is wasteful. Storing rstd lets backward use a multiply.
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
 *   For grad_input we need dL/dx_c, which via chain rule through x̂:
 *   dL/dx_c = rstd * [ dL/dx̂_c
 *                     - (1/C) * Σ_c' dL/dx̂_c'
 *                     - (1/C) * x̂_c * Σ_c' dL/dx̂_c' * x̂_c' ]
 *
 *   where  dL/dx̂_c = dy_c * γ_c
 *
 *   Defining:
 *     sum1 = Σ_c (dy_c * γ_c)
 *     sum2 = Σ_c (dy_c * γ_c * x̂_c)  = Σ_c (dy_c * γ_c * (x_c - μ) * rstd)
 *
 *   The final gradient per input element:
 *     dL/dx_c = rstd * [ dy_c*γ_c - (1/C)*sum1 - (1/C)*(x_c-μ)*rstd²*sum2 ]
 *
 *   This is what lines ~119-129 implement.
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>


// ─── FORWARD PASS ─────────────────────────────────────────────────────────────
/*
 * Returns a tuple of three tensors:
 *   output [N, C]  — normalized + affine output
 *   mean   [N]     — per-sample mean, saved for backward
 *   rstd   [N]     — per-sample reciprocal std, saved for backward
 *
 * Saving mean and rstd avoids recomputing them in backward (they're cheap
 * to store: just 2*N floats vs the N*C input tensor).
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

    auto out  = torch::empty_like(input);         // [N, C] output
    auto mean = torch::empty({N}, input.options()); // [N] mean per sample
    auto rstd = torch::empty({N}, input.options()); // [N] 1/std per sample

    const float* __restrict__ inp = input.data_ptr<float>();
    const float* __restrict__ w   = weight.data_ptr<float>();
    const float* __restrict__ b   = bias.data_ptr<float>();
    float*       __restrict__ o   = out.data_ptr<float>();
    float*       __restrict__ m   = mean.data_ptr<float>();  // output: mean
    float*       __restrict__ r   = rstd.data_ptr<float>();  // output: rstd

    // ── Parallel loop: each sample i is independent ───────────────────────────
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float* row = inp + i * C;  // pointer to input sample i
            float*       dst = o   + i * C;  // pointer to output sample i

            // ── Pass 1: Compute mean μ ────────────────────────────────────────
            // Use double accumulation to avoid float32 catastrophic cancellation
            // when summing many small numbers. For C=2048 with values ~0.1,
            // the sum ~204 but intermediate rounding errors can accumulate.
            double sum = 0.0;
            for (int64_t c = 0; c < C; ++c) sum += row[c];
            float mu = static_cast<float>(sum / C);

            // ── Pass 2: Compute variance σ² ───────────────────────────────────
            // var = (1/C) * Σ (x_c - μ)²
            // Using the centered formula (subtracting mean first) is numerically
            // more stable than the Σx² - (Σx)²/n formula.
            double var_sum = 0.0;
            for (int64_t c = 0; c < C; ++c) {
                float diff = row[c] - mu;
                var_sum += diff * diff;
            }
            // rstd = 1 / sqrt(σ² + ε)
            // Adding ε BEFORE the sqrt ensures we never compute sqrt(negative) due
            // to floating point errors, and prevents division by zero.
            float rs = 1.0f / std::sqrt(static_cast<float>(var_sum / C) + eps);

            // Save statistics for backward pass
            m[i] = mu;
            r[i] = rs;

            // ── Pass 3: Normalize + affine transform ───────────────────────────
            // Fused: (x_c - μ) * rstd * γ_c + β_c
            // No intermediate tensor written — result goes straight to dst.
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
 * This creates "cross terms" in the gradient — the backward is more complex
 * than just multiplying by rstd.
 *
 * See the derivation in the file header comment above for the full math.
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

    auto grad_input  = torch::empty_like(input);          // [N, C]
    auto grad_weight = torch::zeros({C}, input.options()); // [C]  — zero-init, we accumulate
    auto grad_bias   = torch::zeros({C}, input.options()); // [C]  — zero-init, we accumulate

    const float* __restrict__ go  = grad_out.data_ptr<float>();
    const float* __restrict__ inp = input.data_ptr<float>();
    const float* __restrict__ w   = weight.data_ptr<float>();
    const float* __restrict__ mu  = mean.data_ptr<float>();
    const float* __restrict__ rs  = rstd.data_ptr<float>();
    float*       __restrict__ gi  = grad_input.data_ptr<float>();
    float*       __restrict__ gw  = grad_weight.data_ptr<float>();
    float*       __restrict__ gb  = grad_bias.data_ptr<float>();

    // ── Accumulate grad_weight and grad_bias across all N samples ─────────────
    // Sequential loop (not parallelized) to avoid atomic/mutex overhead on gw/gb.
    // For typical batch sizes (≤1024) this is fast enough.
    //
    // dL/dγ_c = Σ_i  grad_out[i,c] * x̂[i,c]   where x̂[i,c] = (x[i,c] - μ[i]) * rstd[i]
    // dL/dβ_c = Σ_i  grad_out[i,c]
    for (int64_t i = 0; i < N; ++i) {
        const float* row_go  = go  + i * C;  // grad_out row i
        const float* row_inp = inp + i * C;  // input row i
        float mi = mu[i], ri = rs[i];        // this sample's mean and rstd

        for (int64_t c = 0; c < C; ++c) {
            // x̂[i,c]: the normalized input before affine transform
            float x_hat = (row_inp[c] - mi) * ri;
            gw[c] += row_go[c] * x_hat;  // accumulate γ gradient
            gb[c] += row_go[c];           // accumulate β gradient
        }
    }

    // ── Compute grad_input (parallelizable: each row is independent) ──────────
    // Each sample's grad_input depends only on its own row of grad_out and input.
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float* row_go  = go  + i * C;
            const float* row_inp = inp + i * C;
            float*       row_gi  = gi  + i * C;
            float mi = mu[i], ri = rs[i];

            // Compute the two aggregate sums needed for the cross-terms.
            // sum1 = Σ_c (dy_c * γ_c)               — "mean of scaled grads"
            // sum2 = Σ_c (dy_c * γ_c * (x_c - μ))   — "covariance of grads with x"
            //
            // These sums come from differentiating through how the mean and
            // variance depend on all inputs. Without them, we'd ignore that
            // "nudging x_c changes μ and σ², which changes ALL outputs".
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                float dy_w  = row_go[c] * w[c];          // dL/dy_c * γ_c = dL/dx̂_c
                sum1 += dy_w;
                sum2 += dy_w * (row_inp[c] - mi);        // × (x_c - μ)
            }

            float inv_C = 1.0f / static_cast<float>(C); // 1/C, hoisted out of loop

            // Apply the LayerNorm backward formula for each element:
            //   dL/dx_c = rstd * [ dL/dx̂_c
            //                     - (1/C) * sum1             (remove mean-gradient)
            //                     - (1/C) * (x_c-μ)*rstd²*sum2 (remove var-gradient) ]
            //
            // The -(1/C)*sum1 term corrects for the fact that changing x_c shifts
            // the mean, which shifts ALL x̂_c' by -(1/C)*rstd.
            //
            // The -(1/C)*(x_c-μ)*rstd²*sum2 term corrects for the fact that
            // changing x_c also shifts the variance, scaling ALL x̂_c'.
            for (int64_t c = 0; c < C; ++c) {
                float x_mu   = row_inp[c] - mi;          // x_c - μ
                float dy_w   = row_go[c] * w[c];         // dL/dx̂_c
                row_gi[c] = ri * (dy_w
                                  - inv_C * sum1
                                  - inv_C * x_mu * ri * ri * sum2);
                // ri*ri = rstd² = 1/(σ²+ε)
                // The term x_mu * ri * ri * sum2 / C is the variance-correction.
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
