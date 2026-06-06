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

// Platform-specific SIMD.
// On Apple Silicon (AArch64 NEON), we use explicit 4-way-unrolled intrinsics
// to eliminate reduction dependency chains.  On x86, the auto-vectorized path
// below is used (the compiler generates SSE/AVX from the scalar loops).
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif


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
            const float* row = inp + i * C;
            float*       dst = o   + i * C;

            // ── Pass 1: fused mean + variance ─────────────────────────────────
            //
            // ROOT CAUSE OF THE (1024,2048) SLOWNESS:
            // Auto-vectorized reductions use a SINGLE accumulator register:
            //   vsum = vsum + vload(row+c)    ← each iteration depends on previous
            // NEON VADD has 3-4 cycle latency, so the CPU stalls waiting for the
            // result before starting the next iteration. The loop is LATENCY-BOUND.
            //
            // Fix: 4 independent accumulator registers (s0,s1,s2,s3) + (q0,q1,q2,q3).
            // Now iterations over s0 have NO dependency on s1/s2/s3, so all 4 chains
            // advance simultaneously. With 2 NEON execute units, we fully saturate
            // both at 2 ops/cycle instead of 1 op per 4 cycles.
            // Speedup on the reduction: ~3-4x.
            float sum = 0.0f, sq_sum = 0.0f;

#ifdef __ARM_NEON
            // 4-way unrolled NEON: 16 floats per outer iteration, 8 independent registers.
            {
                float32x4_t s0=vdupq_n_f32(0), s1=vdupq_n_f32(0),
                            s2=vdupq_n_f32(0), s3=vdupq_n_f32(0);
                float32x4_t q0=vdupq_n_f32(0), q1=vdupq_n_f32(0),
                            q2=vdupq_n_f32(0), q3=vdupq_n_f32(0);

                int64_t c = 0;
                for (; c + 16 <= C; c += 16) {
                    // Load 16 consecutive floats into 4 NEON registers (4 floats each)
                    float32x4_t x0 = vld1q_f32(row + c);
                    float32x4_t x1 = vld1q_f32(row + c + 4);
                    float32x4_t x2 = vld1q_f32(row + c + 8);
                    float32x4_t x3 = vld1q_f32(row + c + 12);
                    // Accumulate sum: 4 independent chains → no stall
                    s0 = vaddq_f32(s0, x0);  s1 = vaddq_f32(s1, x1);
                    s2 = vaddq_f32(s2, x2);  s3 = vaddq_f32(s3, x3);
                    // Accumulate sum-of-squares via FMA: q += x*x
                    q0 = vfmaq_f32(q0, x0, x0);  q1 = vfmaq_f32(q1, x1, x1);
                    q2 = vfmaq_f32(q2, x2, x2);  q3 = vfmaq_f32(q3, x3, x3);
                }
                // Tree-reduce 4 → 2 → 1 NEON register, then horizontal sum
                s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
                q0 = vaddq_f32(vaddq_f32(q0, q1), vaddq_f32(q2, q3));
                sum    = vaddvq_f32(s0);  // 4-lane horizontal sum (AArch64)
                sq_sum = vaddvq_f32(q0);
                // Scalar cleanup for remaining elements (C not multiple of 16)
                for (; c < C; ++c) { sum += row[c]; sq_sum += row[c] * row[c]; }
            }
#else
            // Portable auto-vectorized path (x86 SSE/AVX, non-NEON ARM)
            for (int64_t c = 0; c < C; ++c) {
                float x = row[c]; sum += x; sq_sum += x * x;
            }
#endif

            float mu  = sum * inv_C;
            // Var(X) = E[X²] - E[X]².  std::max(0) guards against -ε rounding.
            float var = std::max(0.0f, sq_sum * inv_C - mu * mu);
            float rs  = 1.0f / std::sqrt(var + eps);
            m[i] = mu;
            r[i] = rs;

            // ── Pass 2: normalize + affine ────────────────────────────────────
            // Same 4-way unrolling for the normalize loop.
            // Each of the 4 groups (x0,w0,b0), (x1,w1,b1), ... is independent,
            // so the 4 FMA computations can all be in flight simultaneously.

#ifdef __ARM_NEON
            {
                float32x4_t vrs        = vdupq_n_f32(rs);
                // Key optimization: replace vsub(x,mu) + vmul(result,rs) with a
                // single FMA using the identity: (x - mu)*rs = x*rs + (-mu*rs).
                //
                // Old: n = vmulq(vsubq(x, vmu), vrs)   ← VSUB + VMUL, 2 instructions,
                //           vsubq must complete before vmulq can start (2-cycle chain)
                //
                // New: n = vfmaq(vneg_murs, x, vrs)    ← single VFMA instruction
                //   where vneg_murs = -mu * rs  (precomputed once, outside loop)
                //   vfmaq(a, b, c) = a + b*c
                //   so: vneg_murs + x*rs = -mu*rs + x*rs = (x-mu)*rs  ✓
                //
                // Saves 4 instructions per 16 elements (VSUB eliminated for all 4 groups).
                // For C=2048: saves 2048/16 × 4 = 512 arithmetic instructions per row.
                float32x4_t vneg_murs  = vdupq_n_f32(-mu * rs);  // -mu*rs, same all cols

                int64_t c = 0;
                for (; c + 16 <= C; c += 16) {
                    // Load input, weight, bias for 16 elements (4 groups of 4)
                    float32x4_t x0=vld1q_f32(row+c),   x1=vld1q_f32(row+c+4),
                                x2=vld1q_f32(row+c+8),  x3=vld1q_f32(row+c+12);
                    float32x4_t w0=vld1q_f32(w+c),     w1=vld1q_f32(w+c+4),
                                w2=vld1q_f32(w+c+8),    w3=vld1q_f32(w+c+12);
                    float32x4_t b0=vld1q_f32(b+c),     b1=vld1q_f32(b+c+4),
                                b2=vld1q_f32(b+c+8),    b3=vld1q_f32(b+c+12);
                    // n_i = x_i * rs + (-mu*rs) = (x_i - mu)*rs   (single FMA)
                    float32x4_t n0=vfmaq_f32(vneg_murs, x0, vrs);
                    float32x4_t n1=vfmaq_f32(vneg_murs, x1, vrs);
                    float32x4_t n2=vfmaq_f32(vneg_murs, x2, vrs);
                    float32x4_t n3=vfmaq_f32(vneg_murs, x3, vrs);
                    // y_i = b_i + n_i * w_i   (second FMA)
                    vst1q_f32(dst+c,    vfmaq_f32(b0, n0, w0));
                    vst1q_f32(dst+c+4,  vfmaq_f32(b1, n1, w1));
                    vst1q_f32(dst+c+8,  vfmaq_f32(b2, n2, w2));
                    vst1q_f32(dst+c+12, vfmaq_f32(b3, n3, w3));
                }
                // Scalar cleanup for C not a multiple of 16
                for (; c < C; ++c) dst[c] = (row[c] - mu) * rs * w[c] + b[c];
            }
#else
            for (int64_t c = 0; c < C; ++c) {
                dst[c] = (row[c] - mu) * rs * w[c] + b[c];
            }
#endif
        }
    });

    return {out, mean, rstd};
}


// ─── INFERENCE FORWARD (no mean/rstd output) ─────────────────────────────────
/*
 * Faster path for inference (torch.no_grad()) — returns only the output tensor.
 *
 * The training forward allocates and fills mean[N] and rstd[N], saves them as
 * output tensors, then returns a 3-tuple. In Python, no_grad() discards them:
 *   out, _, _ = ext.forward(x, w, b, eps)   ← 2 wasted allocations + writes + unpacks
 *
 * This function:
 *   - Skips allocating mean and rstd entirely (saves 2 × N × 4 bytes of allocation)
 *   - Returns a single tensor (no Python tuple overhead)
 *   - Uses the identical NEON/scalar inner loops as training forward
 *
 * For N=1024: saves ~8KB of writes + 2 tensor allocations + Python tuple unpack.
 * Small savings individually, but adds up over many calls in inference loops.
 */
torch::Tensor layer_norm_inference(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps)
{
    TORCH_CHECK(input.dim() == 2,   "input must be 2D");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(),   "bias must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const float inv_C = 1.0f / static_cast<float>(C);

    auto out = torch::empty_like(input);

    const float* __restrict__ inp = input.data_ptr<float>();
    const float* __restrict__ w   = weight.data_ptr<float>();
    const float* __restrict__ b   = bias.data_ptr<float>();
    float*       __restrict__ o   = out.data_ptr<float>();

    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float* row = inp + i * C;
            float*       dst = o   + i * C;

            float sum = 0.0f, sq_sum = 0.0f;
#ifdef __ARM_NEON
            {
                float32x4_t s0=vdupq_n_f32(0), s1=vdupq_n_f32(0),
                            s2=vdupq_n_f32(0), s3=vdupq_n_f32(0);
                float32x4_t q0=vdupq_n_f32(0), q1=vdupq_n_f32(0),
                            q2=vdupq_n_f32(0), q3=vdupq_n_f32(0);
                int64_t c = 0;
                for (; c + 16 <= C; c += 16) {
                    float32x4_t x0=vld1q_f32(row+c),   x1=vld1q_f32(row+c+4),
                                x2=vld1q_f32(row+c+8),  x3=vld1q_f32(row+c+12);
                    s0=vaddq_f32(s0,x0); s1=vaddq_f32(s1,x1);
                    s2=vaddq_f32(s2,x2); s3=vaddq_f32(s3,x3);
                    q0=vfmaq_f32(q0,x0,x0); q1=vfmaq_f32(q1,x1,x1);
                    q2=vfmaq_f32(q2,x2,x2); q3=vfmaq_f32(q3,x3,x3);
                }
                s0=vaddq_f32(vaddq_f32(s0,s1),vaddq_f32(s2,s3));
                q0=vaddq_f32(vaddq_f32(q0,q1),vaddq_f32(q2,q3));
                sum=vaddvq_f32(s0); sq_sum=vaddvq_f32(q0);
                for (; c < C; ++c) { sum+=row[c]; sq_sum+=row[c]*row[c]; }
            }
#else
            for (int64_t c = 0; c < C; ++c) { float x=row[c]; sum+=x; sq_sum+=x*x; }
#endif
            float mu  = sum * inv_C;
            float var = std::max(0.0f, sq_sum * inv_C - mu * mu);
            float rs  = 1.0f / std::sqrt(var + eps);
            // No m[i]/r[i] writes — that's the whole point of this function

#ifdef __ARM_NEON
            {
                float32x4_t vrs       = vdupq_n_f32(rs);
                float32x4_t vneg_murs = vdupq_n_f32(-mu * rs);
                int64_t c = 0;
                for (; c + 16 <= C; c += 16) {
                    float32x4_t x0=vld1q_f32(row+c),   x1=vld1q_f32(row+c+4),
                                x2=vld1q_f32(row+c+8),  x3=vld1q_f32(row+c+12);
                    float32x4_t w0=vld1q_f32(w+c),     w1=vld1q_f32(w+c+4),
                                w2=vld1q_f32(w+c+8),    w3=vld1q_f32(w+c+12);
                    float32x4_t b0=vld1q_f32(b+c),     b1=vld1q_f32(b+c+4),
                                b2=vld1q_f32(b+c+8),    b3=vld1q_f32(b+c+12);
                    float32x4_t n0=vfmaq_f32(vneg_murs,x0,vrs);
                    float32x4_t n1=vfmaq_f32(vneg_murs,x1,vrs);
                    float32x4_t n2=vfmaq_f32(vneg_murs,x2,vrs);
                    float32x4_t n3=vfmaq_f32(vneg_murs,x3,vrs);
                    vst1q_f32(dst+c,    vfmaq_f32(b0,n0,w0));
                    vst1q_f32(dst+c+4,  vfmaq_f32(b1,n1,w1));
                    vst1q_f32(dst+c+8,  vfmaq_f32(b2,n2,w2));
                    vst1q_f32(dst+c+12, vfmaq_f32(b3,n3,w3));
                }
                for (; c < C; ++c) dst[c] = (row[c]-mu)*rs*w[c]+b[c];
            }
#else
            for (int64_t c = 0; c < C; ++c) dst[c] = (row[c]-mu)*rs*w[c]+b[c];
#endif
        }
    });
    return out;
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
    m.def("forward",           &layer_norm_forward,    "LayerNorm training forward (returns output + mean + rstd)");
    m.def("inference_forward", &layer_norm_inference,  "LayerNorm inference forward (output only, no mean/rstd)");
    m.def("backward",          &layer_norm_backward,   "LayerNorm backward");
}
