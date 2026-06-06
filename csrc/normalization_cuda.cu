/*
 * normalization_cuda.cu — GPU Layer Normalization Kernel
 *
 * WHY GPU IS DRAMATICALLY FASTER FOR LAYERNORM
 * ─────────────────────────────────────────────
 * The CPU bottleneck is memory bandwidth (~50-200 GB/s) and the dependency
 * chains in reduction loops that stall SIMD execution.
 *
 * GPU advantages:
 *   Memory bandwidth: 900 GB/s (A100), 3.35 TB/s (H100) — 10-20× faster
 *   Parallelism: N rows → N CUDA blocks, all running simultaneously
 *   rsqrtf(): single hardware instruction (FRSQRT), ~4 cycles vs div+sqrt on CPU
 *
 * GPU ARCHITECTURE CONCEPTS USED HERE
 * ─────────────────────────────────────
 * Thread:  one lane of computation, executes one instruction stream
 * Warp:    32 threads that execute in lockstep (SIMT — Single Instruction, Multiple Thread)
 * Block:   group of warps sharing L1/shared memory, assigned to one SM
 * Grid:    all blocks; one block per input row here
 *
 * Memory hierarchy (A100, fastest to slowest):
 *   Registers      256 KB/SM    ~0 cycles      private per-thread
 *   Shared memory  48-164 KB/SM ~20 cycles     shared within a block
 *   L2 cache       40 MB        ~200 cycles    shared across SMs
 *   HBM (GPU RAM)  80 GB        ~800 cycles    2 TB/s bandwidth
 *
 * KEY TECHNIQUE: WARP SHUFFLE REDUCTION
 * ──────────────────────────────────────
 * Classic reduction: each warp uses shared memory.
 *   Thread → write to shared[tid] → __syncwarp() → read neighbor → add
 *   Cost: ~20 cycles per level × 5 levels = ~100 cycles
 *
 * Warp shuffle (__shfl_down_sync): threads exchange REGISTERS directly.
 *   No shared memory, no sync overhead, just 1 cycle per step.
 *   __shfl_down_sync(mask, v, offset) makes every thread read from
 *   the thread `offset` positions ahead in the same warp.
 *
 *   Butterfly pattern (5 steps for 32→1):
 *   offset=16: t[0]+=t[16], t[1]+=t[17], ..., t[15]+=t[31]
 *   offset=8:  t[0]+=t[8],  t[1]+=t[9],  ..., t[7]+=t[15]
 *   offset=4:  t[0]+=t[4],  t[1]+=t[5],  t[2]+=t[6], t[3]+=t[7]
 *   offset=2:  t[0]+=t[2],  t[1]+=t[3]
 *   offset=1:  t[0]+=t[1]
 *   → t[0] has the sum of all 32 values. Cost: 5 cycles.
 *
 * KERNEL STRUCTURE: one CUDA block per input row
 * ───────────────────────────────────────────────
 * With block_size=256 threads and C=2048 features:
 *   Each thread handles C/256 = 8 features
 *   256 threads / 32 (warp size) = 8 warps per block
 *
 * Phase 1: Each thread accumulates partial sum over its assigned features (no sync)
 * Phase 2: Warp reduction (32→1) via __shfl_down_sync (no shared memory)
 * Phase 3: Cross-warp reduction (8 warp sums → 1) via shared memory (1 syncthreads)
 * Phase 4: Thread 0 computes mu/rstd, writes to shared scalars
 * Phase 5: All threads read mu/rstd, normalize their features (no sync after)
 *
 * Total __syncthreads() calls: 2. Minimal synchronization overhead.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>


// ─── Warp-level parallel sum reduction ───────────────────────────────────────
/*
 * Reduces 32 values across a warp to a single sum in lane 0.
 * Uses register-to-register communication — no shared memory, no bank conflicts.
 *
 * __shfl_down_sync(mask, value, offset):
 *   Each thread receives the value from the thread 'offset' lanes ahead.
 *   0xffffffff mask = all 32 threads participate.
 *   After 5 steps, lane 0 holds the sum of all 32 original values.
 */
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;  // valid only in lane 0
}


// ─── Block-level reduction helper ────────────────────────────────────────────
/*
 * Takes a per-thread value, reduces within each warp, then across warps.
 * Returns the block total in thread 0 (other threads have undefined value).
 *
 * s_scratch must be __shared__ float[32] — one slot per possible warp.
 * For blockDim.x <= 1024, n_warps <= 32.
 */
__device__ __forceinline__ float block_reduce_sum(
    float v, float* __restrict__ s_scratch)
{
    int lane_id = threadIdx.x & 31;        // position within warp
    int warp_id = threadIdx.x >> 5;        // which warp
    int n_warps = (blockDim.x + 31) >> 5;  // total warps in block

    // Warp reduction: lane 0 of each warp holds warp sum
    v = warp_reduce_sum(v);

    // Write each warp's sum to shared memory
    if (lane_id == 0) s_scratch[warp_id] = v;
    __syncthreads();

    // First warp reads all warp sums and reduces them
    v = (lane_id < n_warps) ? s_scratch[lane_id] : 0.f;
    if (warp_id == 0) v = warp_reduce_sum(v);

    return v;  // valid only in thread 0
}


// ─── FORWARD KERNEL ──────────────────────────────────────────────────────────
/*
 * One block processes one input row.
 * Computes: y[i, c] = (x[i, c] - μᵢ) / √(σᵢ² + ε) * γ[c] + β[c]
 *
 * Grid:  (N,)     — one block per sample
 * Block: (T,)     — T threads share the work for C features
 */
__global__ void layer_norm_fwd_kernel(
    const float* __restrict__ inp,      // [N, C] input
    const float* __restrict__ weight,   // [C]    gamma (affine scale)
    const float* __restrict__ bias,     // [C]    beta (affine shift)
    float*       __restrict__ out,      // [N, C] output
    float*       __restrict__ mean_out, // [N]    saved mean
    float*       __restrict__ rstd_out, // [N]    saved reciprocal std
    int C, float eps)
{
    int row = blockIdx.x;   // which row (sample) this block handles
    int tid = threadIdx.x;  // this thread's index within the block
    int T   = blockDim.x;   // total threads in block

    const float* x = inp + (long)row * C;
    float*       y = out + (long)row * C;

    // ── Phase 1: Each thread accumulates over its strided slice of C ──────────
    // Thread 0 processes c = 0, T, 2T, ...
    // Thread 1 processes c = 1, T+1, 2T+1, ...
    // → consecutive threads access consecutive memory = COALESCED ACCESS
    // Coalesced access: the GPU can merge the 32 threads in a warp into one
    // wide memory transaction (128 bytes = 32 floats), maximizing bandwidth.
    float tsum = 0.f, tssq = 0.f;
    for (int c = tid; c < C; c += T) {
        float v = x[c];
        tsum += v;
        tssq += v * v;  // computing E[X²] alongside E[X] in one pass
    }

    // ── Phase 2+3: Full block reduction of tsum and tssq ─────────────────────
    // Uses warp shuffles then shared memory. Total cost: 2 syncthreads.
    __shared__ float s_buf[32];
    tsum = block_reduce_sum(tsum, s_buf);
    __syncthreads();  // reuse s_buf for tssq
    tssq = block_reduce_sum(tssq, s_buf);

    // ── Phase 4: Thread 0 computes μ and rstd, broadcasts via shared mem ──────
    __shared__ float s_mu, s_rs;
    if (tid == 0) {
        float mu  = tsum / C;
        // Var(X) = E[X²] - E[X]²  (computationally equivalent to centered formula)
        // fmaxf(., 0) guards against -ε from floating-point rounding.
        float var = fmaxf(tssq / C - mu * mu, 0.f);
        // rsqrtf: hardware reciprocal square root (~4 cycles on all NVIDIA GPUs).
        // 1/sqrt(x) via div+sqrt takes ~20-40 cycles. Always prefer rsqrtf.
        float rs  = rsqrtf(var + eps);
        s_mu = mu;
        s_rs = rs;
        mean_out[row] = mu;   // saved for backward
        rstd_out[row] = rs;   // saved for backward
    }
    __syncthreads();  // all threads must see s_mu/s_rs before normalizing
    float mu = s_mu, rs = s_rs;

    // ── Phase 5: Normalize and apply affine transform ────────────────────────
    // y[c] = (x[c] - μ) * rstd * γ[c] + β[c]
    // Stride access, same coalescing pattern as Phase 1.
    for (int c = tid; c < C; c += T) {
        y[c] = (x[c] - mu) * rs * weight[c] + bias[c];
    }
}


// ─── BACKWARD: grad_input KERNEL ─────────────────────────────────────────────
/*
 * Computes dL/dx for one row.
 * Reuses the block reduction pattern from the forward kernel.
 *
 * The LayerNorm backward formula (derived via chain rule through shared μ/σ):
 *   Define: dy_w[c] = dy[c] * γ[c]    (upstream grad scaled by weight)
 *           sum1    = Σ_c dy_w[c]
 *           sum2    = Σ_c dy_w[c] * (x[c] - μ)
 *
 *   dx[c] = rstd * [ dy_w[c]
 *                   - (1/C)*sum1               ← corrects for mean's dependency on x
 *                   - (1/C)*(x[c]-μ)*rstd²*sum2 ← corrects for variance's dependency on x ]
 *
 * The correction terms exist because nudging any single x[c] changes μ AND σ²,
 * which then shifts ALL normalized outputs. A naive elementwise backward would be wrong.
 */
__global__ void layer_norm_bwd_dx_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float*       __restrict__ dx,
    int C)
{
    int row = blockIdx.x, tid = threadIdx.x, T = blockDim.x;

    const float* dy_r = dy + (long)row * C;
    const float* x_r  = x  + (long)row * C;
    float*       dx_r = dx + (long)row * C;
    float mu = mean[row], rs = rstd[row];

    // Pass 1: compute sum1 and sum2
    float sum1 = 0.f, sum2 = 0.f;
    for (int c = tid; c < C; c += T) {
        float dyw = dy_r[c] * weight[c];
        sum1 += dyw;
        sum2 += dyw * (x_r[c] - mu);
    }

    __shared__ float s_buf[32];
    sum1 = block_reduce_sum(sum1, s_buf);
    __syncthreads();
    sum2 = block_reduce_sum(sum2, s_buf);

    __shared__ float ss1, ss2;
    if (tid == 0) { ss1 = sum1; ss2 = sum2; }
    __syncthreads();
    sum1 = ss1; sum2 = ss2;

    // Pass 2: apply backward formula
    float inv_C = 1.f / C;
    float rs2   = rs * rs;  // rstd² = 1/(σ²+ε)
    for (int c = tid; c < C; c += T) {
        float dyw  = dy_r[c] * weight[c];
        float x_mu = x_r[c] - mu;
        dx_r[c] = rs * (dyw - inv_C * sum1 - inv_C * x_mu * rs2 * sum2);
    }
}


// ─── BACKWARD: grad_weight and grad_bias KERNEL ───────────────────────────────
/*
 * Computes dL/dγ and dL/dβ.
 * These require summing over ALL N rows (unlike grad_input which is per-row).
 *
 * Strategy: one block per COLUMN (feature), reducing over N rows.
 *   Grid:  (C,)  — one block per feature
 *   Block: (T,)  — T threads divide N rows
 *
 * dL/dγ[c] = Σ_i  dy[i,c] * x̂[i,c]    where x̂[i,c] = (x[i,c] - μᵢ) * rstdᵢ
 * dL/dβ[c] = Σ_i  dy[i,c]
 */
__global__ void layer_norm_bwd_gb_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float*       __restrict__ dgamma,
    float*       __restrict__ dbeta,
    int N, int C)
{
    int c = blockIdx.x, tid = threadIdx.x, T = blockDim.x;

    float dg = 0.f, db = 0.f;
    for (int row = tid; row < N; row += T) {
        // x̂[row,c] = (x[row,c] - μ[row]) * rstd[row]
        float xhat = (x[(long)row * C + c] - mean[row]) * rstd[row];
        dg += dy[(long)row * C + c] * xhat;
        db += dy[(long)row * C + c];
    }

    __shared__ float s_buf[32];
    dg = block_reduce_sum(dg, s_buf);
    __syncthreads();
    db = block_reduce_sum(db, s_buf);

    if (tid == 0) {
        dgamma[c] = dg;
        dbeta[c]  = db;
    }
}


// ─── C++ HOST WRAPPERS ───────────────────────────────────────────────────────
/*
 * These are called from Python. They:
 *   1. Validate tensors
 *   2. Choose block size
 *   3. Launch CUDA kernels on the current CUDA stream
 *   4. Return output tensors (CUDA kernels execute asynchronously)
 *
 * at::cuda::getCurrentCUDAStream() returns the stream associated with the
 * current PyTorch context — ensures our kernel runs in-order with other ops.
 */

static int choose_block_size(int C) {
    // More threads = better parallelism for large C.
    // Fewer threads = fewer idle threads for small C.
    if (C >= 512) return 256;
    if (C >= 128) return 128;
    if (C >= 32)  return 64;
    return 32;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps)
{
    TORCH_CHECK(input.is_cuda(),   "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),  "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),    "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2,  "input must be 2D [batch, features]");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(),   "bias must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    const int N = input.size(0);
    const int C = input.size(1);
    TORCH_CHECK(weight.size(0) == C, "weight size must match features");
    TORCH_CHECK(bias.size(0)   == C, "bias size must match features");

    auto output   = torch::empty_like(input);
    auto mean_out = torch::empty({N}, input.options());
    auto rstd_out = torch::empty({N}, input.options());

    int block = choose_block_size(C);

    // Grid = (N,): one block per row. Block = (block,): threads per block.
    layer_norm_fwd_kernel<<<N, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        rstd_out.data_ptr<float>(),
        C, eps);

    return {output, mean_out, rstd_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    float /*eps*/)
{
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(input.is_contiguous(),    "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),   "weight must be contiguous");

    const int N = input.size(0);
    const int C = input.size(1);

    auto dx     = torch::empty_like(input);
    auto dgamma = torch::empty({C}, input.options());
    auto dbeta  = torch::empty({C}, input.options());

    int block  = choose_block_size(C);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Kernel 1: per-row grad_input (grid = N blocks)
    layer_norm_bwd_dx_kernel<<<N, block, 0, stream>>>(
        grad_out.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        dx.data_ptr<float>(),
        C);

    // Kernel 2: per-column grad_weight and grad_bias (grid = C blocks)
    layer_norm_bwd_gb_kernel<<<C, block, 0, stream>>>(
        grad_out.data_ptr<float>(),
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        dgamma.data_ptr<float>(),
        dbeta.data_ptr<float>(),
        N, C);

    return {dx, dgamma, dbeta};
}


// ─── Python bindings ─────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &layer_norm_cuda_forward,  "LayerNorm CUDA forward");
    m.def("backward", &layer_norm_cuda_backward, "LayerNorm CUDA backward");
}
