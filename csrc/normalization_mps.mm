/*
 * normalization_mps.mm — Custom Metal LayerNorm for Apple Silicon GPU
 *
 * FIX HISTORY:
 *   v1: crashed — temporary tensor destroyed before GPU read it (use-after-free)
 *   v2: crashed — commandEncoder() corrupted PyTorch's active encoder state
 *   v3: aborted  — no @autoreleasepool, Metal's internal autoreleased objects
 *                  had nowhere to register, corrupted the pool page
 *   v4: segfault — newBufferWithBytesNoCopy called with non-page-aligned length
 *
 * ROOT CAUSE OF v4 CRASH:
 * ────────────────────────
 * On Apple Silicon, newBufferWithBytesNoCopy requires BOTH the pointer AND the
 * byte length to be multiples of the VM page size (16 384 bytes).
 *
 * Our small tensors violate the length requirement:
 *   mean_out  = N×4 bytes  →  N=32: 128 bytes, N=1024: 4 096 bytes  ✗
 *   weight    = C×4 bytes  →  C=2048: 8 192 bytes                    ✗
 *   bias      = C×4 bytes  →  same                                    ✗
 *
 * Metal does NOT gracefully return nil for bad length — it crashes.
 *
 * FIX:
 *   Use newBufferWithBytesNoCopy  when  len % 16384 == 0  (large I/O tensors)
 *   Use newBufferWithBytes:       when  len % 16384 != 0  (small tensors)
 *     newBufferWithBytes: copies data into a Metal-owned, page-aligned buffer.
 *     On Apple Silicon unified memory this is a ~CPU-speed memcpy,
 *     and the small tensors (max 8 KB) make it negligible (~0.2 μs).
 *
 * For the main input/output (N×C floats):
 *   (32, 256):   32 768 bytes  = 2 × 16 384  ✓  zero-copy
 *   (256, 1024): 1 048 576 bytes              ✓  zero-copy
 *   (1024, 2048): 8 388 608 bytes             ✓  zero-copy
 */

#include <torch/extension.h>
#import <Metal/Metal.h>
#include <mutex>


// ─── Metal shader ──────────────────────────────────────────────────────────────
static const char* kShaderSrc = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_fwd(
    device const float* inp     [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    device float*       mu_out  [[buffer(4)]],
    device float*       rs_out  [[buffer(5)]],
    constant uint&      C_val   [[buffer(6)]],
    constant float&     eps_val [[buffer(7)]],
    threadgroup float*  smem    [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint T   [[threads_per_threadgroup]])
{
    const device float* x = inp + (ulong)row * C_val;
    device float*       y = out + (ulong)row * C_val;

    float tsum = 0.f, tssq = 0.f;
    for (uint c = tid; c < C_val; c += T) {
        float v = x[c];
        tsum += v;
        tssq += v * v;
    }

    tsum = simd_sum(tsum);
    tssq = simd_sum(tssq);

    uint simd_id = tid >> 5, lane_id = tid & 31;
    uint n_simds = (T + 31) >> 5;
    threadgroup float* s_sum = smem;
    threadgroup float* s_ssq = smem + n_simds;
    if (lane_id == 0) { s_sum[simd_id] = tsum; s_ssq[simd_id] = tssq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        tsum = (lane_id < n_simds) ? s_sum[lane_id] : 0.f;
        tssq = (lane_id < n_simds) ? s_ssq[lane_id] : 0.f;
        tsum = simd_sum(tsum);
        tssq = simd_sum(tssq);
    }

    threadgroup float s_mu, s_rs;
    if (tid == 0) {
        float mu  = tsum / float(C_val);
        float var = max(0.f, tssq / float(C_val) - mu * mu);
        float rs  = rsqrt(var + eps_val);
        s_mu = mu; s_rs = rs;
        mu_out[row] = mu;
        rs_out[row] = rs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mu = s_mu, rs = s_rs;

    float neg_mu_rs = -mu * rs;
    for (uint c = tid; c < C_val; c += T) {
        y[c] = fma(x[c], rs, neg_mu_rs) * weight[c] + bias[c];
    }
}
)metal";


// ─── Globals ──────────────────────────────────────────────────────────────────
static id<MTLDevice>               g_device;
static id<MTLCommandQueue>         g_queue;
static id<MTLComputePipelineState> g_pso;
static std::once_flag              g_init;

static void metal_init() {
    std::call_once(g_init, []() {
        @autoreleasepool {
            g_device = MTLCreateSystemDefaultDevice();
            TORCH_CHECK(g_device, "No Metal device found");
            g_queue  = [g_device newCommandQueue];
            TORCH_CHECK(g_queue,  "Failed to create MTLCommandQueue");
            NSError* err = nil;
            id<MTLLibrary> lib = [g_device newLibraryWithSource:@(kShaderSrc)
                                                        options:nil error:&err];
            TORCH_CHECK(err == nil,
                "Metal shader compile: ", [[err localizedDescription] UTF8String]);
            id<MTLFunction> fn = [lib newFunctionWithName:@"layer_norm_fwd"];
            TORCH_CHECK(fn, "Metal function not found");
            g_pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
            TORCH_CHECK(err == nil,
                "Metal PSO: ", [[err localizedDescription] UTF8String]);
        }
    });
}


// ─── Buffer helper ────────────────────────────────────────────────────────────
// Apple Silicon: newBufferWithBytesNoCopy requires len % 16384 == 0.
// For large tensors (input / main output) this is satisfied.
// For small tensors (weight, bias, mean, rstd) it is NOT — use copy.
static const NSUInteger kPageSz = 16384;

static id<MTLBuffer> inputBuffer(id<MTLDevice> dev, const torch::Tensor& t) {
    NSUInteger len = (NSUInteger)t.nbytes();
    if (len % kPageSz == 0) {
        // Zero-copy: Metal buffer aliases PyTorch's page-aligned MPS memory.
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:t.data_ptr()
            length:len options:MTLResourceStorageModeShared deallocator:nil];
        if (b) return b;
    }
    // Copy fallback: Metal allocates aligned buffer and copies in.
    // For our small tensors (≤8 KB) the overhead is negligible (~0.2 μs).
    return [dev newBufferWithBytes:t.data_ptr()
            length:len options:MTLResourceStorageModeShared];
}

// For output tensors: if page-aligned use zero-copy, else allocate fresh Metal
// buffer. The bool written to *did_copy tells the caller whether to memcpy back.
static id<MTLBuffer> outputBuffer(id<MTLDevice> dev, const torch::Tensor& t,
                                   bool* did_copy) {
    NSUInteger len = (NSUInteger)t.nbytes();
    if (len % kPageSz == 0) {
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:t.data_ptr()
            length:len options:MTLResourceStorageModeShared deallocator:nil];
        if (b) { *did_copy = false; return b; }
    }
    *did_copy = true;
    return [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
}


// ─── Forward ──────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_mps_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps)
{
    TORCH_CHECK(input.device().type() == torch::kMPS, "input must be on MPS");
    TORCH_CHECK(input.dim() == 2 && input.is_contiguous(), "input: 2D contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    metal_init();

    const int64_t N = input.size(0), C = input.size(1);
    TORCH_CHECK(weight.size(0) == C && bias.size(0) == C);

    // Named locals keep tensors alive past waitUntilCompleted.
    auto inp_c = input;
    auto wgt_c = weight.contiguous().to(torch::kFloat32);
    auto bia_c = bias.contiguous().to(torch::kFloat32);
    auto out      = torch::empty_like(inp_c);
    auto mean_out = torch::empty({N}, inp_c.options());
    auto rstd_out = torch::empty({N}, inp_c.options());

    uint32_t  C_u  = (uint32_t)C;
    NSUInteger T   = (C >= 512) ? 256 : (C >= 128 ? 128 : 64);
    NSUInteger smem = 2 * ((T + 31) / 32) * sizeof(float);

    @autoreleasepool {
        // Build Metal buffers.
        // Large tensors (N×C floats, always multiples of 16384) → zero-copy.
        // Small tensors (C floats for w/b, N floats for stats)  → copy.
        id<MTLBuffer> buf_inp = inputBuffer(g_device, inp_c);
        id<MTLBuffer> buf_wgt = inputBuffer(g_device, wgt_c);
        id<MTLBuffer> buf_bia = inputBuffer(g_device, bia_c);

        bool out_copy, mu_copy, rs_copy;
        id<MTLBuffer> buf_out = outputBuffer(g_device, out,      &out_copy);
        id<MTLBuffer> buf_mu  = outputBuffer(g_device, mean_out, &mu_copy);
        id<MTLBuffer> buf_rs  = outputBuffer(g_device, rstd_out, &rs_copy);

        id<MTLCommandBuffer>        cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pso];
        [enc setBuffer:buf_inp offset:0 atIndex:0];
        [enc setBuffer:buf_wgt offset:0 atIndex:1];
        [enc setBuffer:buf_bia offset:0 atIndex:2];
        [enc setBuffer:buf_out offset:0 atIndex:3];
        [enc setBuffer:buf_mu  offset:0 atIndex:4];
        [enc setBuffer:buf_rs  offset:0 atIndex:5];
        [enc setBytes:&C_u length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&eps  length:sizeof(float)   atIndex:7];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        NSError* e = [cmd error];
        TORCH_CHECK(e == nil, "Metal execution error: ",
                    [[e localizedDescription] UTF8String]);

        // Copy results out of Metal-owned buffers back into PyTorch tensors
        // (only needed for tensors that couldn't use zero-copy).
        if (out_copy) memcpy(out.data_ptr(),      [buf_out contents], out.nbytes());
        if (mu_copy)  memcpy(mean_out.data_ptr(), [buf_mu  contents], mean_out.nbytes());
        if (rs_copy)  memcpy(rstd_out.data_ptr(), [buf_rs  contents], rstd_out.nbytes());
    }

    return {out, mean_out, rstd_out};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layer_norm_mps_forward, "LayerNorm Metal kernel");
}
