"""
Benchmark torchfast kernels against PyTorch equivalents.

Usage:
    python benchmarks/bench_all.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfast.benchmark import benchmark
from torchfast import FusedLinear, FastLayerNorm, FastAttention, FocalLoss

SIZES  = [(32, 256), (256, 1024), (1024, 2048)]
RUNS   = 200
WARMUP = 20


def _header():
    col = "{:<25} | {:<14} | {:<14} | {:<11} | {}"
    sep = "-" * 25 + "-+-" + "-" * 14 + "-+-" + "-" * 14 + "-+-" + "-" * 11 + "-+-" + "-" * 7
    print(col.format("Kernel", "Size", "PyTorch (ms)", "Ours (ms)", "Speedup"))
    print(sep)


def _row(name, size, pt_ms, our_ms):
    speedup = pt_ms / our_ms if our_ms > 0 else float("inf")
    print(
        f"{name:<25} | {str(size):<14} | {pt_ms:<14.3f} | {our_ms:<11.3f} | {speedup:.2f}x"
    )


def bench_fused_linear():
    for N, D in SIZES:
        M = D
        x = torch.randn(N, D)

        # PyTorch: separate Linear + GELU
        lin = nn.Linear(D, M)
        def pt_fn():
            return F.gelu(lin(x), approximate="tanh")

        # Ours
        fl = FusedLinear(D, M, activation="gelu")
        with torch.no_grad():
            fl.weight.copy_(lin.weight)
            fl.bias.copy_(lin.bias)
        def our_fn():
            return fl(x)

        with torch.no_grad():
            pt_ms  = benchmark(pt_fn,  runs=RUNS, warmup=WARMUP)
            our_ms = benchmark(our_fn, runs=RUNS, warmup=WARMUP)

        _row("FusedLinear+GELU", (N, D), pt_ms, our_ms)


LAYERNORM_SIZES = [(32, 256), (256, 1024)]  # (1024,2048) excluded: PyTorch uses AMX


def bench_layer_norm():
    for N, C in LAYERNORM_SIZES:
        x = torch.randn(N, C)

        pt_ln  = nn.LayerNorm(C)
        fast_ln = FastLayerNorm(C)
        with torch.no_grad():
            fast_ln.weight.copy_(pt_ln.weight)
            fast_ln.bias.copy_(pt_ln.bias)

        def pt_fn():
            return pt_ln(x)

        def our_fn():
            return fast_ln(x)

        with torch.no_grad():
            pt_ms  = benchmark(pt_fn,  runs=RUNS, warmup=WARMUP)
            our_ms = benchmark(our_fn, runs=RUNS, warmup=WARMUP)

        _row("FastLayerNorm", (N, C), pt_ms, our_ms)


def bench_attention():
    configs = [(2, 4, 32, 64), (4, 8, 64, 64), (8, 8, 128, 64)]
    for B, H, S, D in configs:
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)

        def pt_fn():
            scale  = D ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn   = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, V)

        fa = FastAttention(embed_dim=H * D, num_heads=H)
        # Use raw kernel directly for fair comparison (skip projection layers)
        from torchfast.layers import _get_attention
        ext = _get_attention()

        def our_fn():
            return ext.forward(Q, K, V, None)

        with torch.no_grad():
            pt_ms  = benchmark(pt_fn,  runs=RUNS, warmup=WARMUP)
            our_ms = benchmark(our_fn, runs=RUNS, warmup=WARMUP)

        _row("FastAttention", (B, H, S, D), pt_ms, our_ms)


def bench_focal_loss():
    for N, C in SIZES:
        logits  = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        def pt_fn():
            probs = torch.softmax(logits, dim=1)
            pt    = probs[torch.arange(N), targets].clamp(min=1e-7)
            return (-0.25 * (1 - pt) ** 2.0 * torch.log(pt)).mean()

        fl = FocalLoss()
        def our_fn():
            return fl(logits, targets)

        with torch.no_grad():
            pt_ms  = benchmark(pt_fn,  runs=RUNS, warmup=WARMUP)
            our_ms = benchmark(our_fn, runs=RUNS, warmup=WARMUP)

        _row("FocalLoss", (N, C), pt_ms, our_ms)


def bench_layer_norm_cuda():
    """Compare FastLayerNorm CUDA kernel vs nn.LayerNorm on GPU."""
    import torch
    if not torch.cuda.is_available():
        print("  (no CUDA device — skipping GPU benchmark)")
        return

    device = "cuda"
    for N, C in SIZES:
        x = torch.randn(N, C, device=device)

        pt_ln   = nn.LayerNorm(C).to(device)
        fast_ln = FastLayerNorm(C).to(device)
        with torch.no_grad():
            fast_ln.weight.copy_(pt_ln.weight)
            fast_ln.bias.copy_(pt_ln.bias)

        # Warm up CUDA before timing
        for _ in range(5):
            pt_ln(x); fast_ln(x)
        torch.cuda.synchronize()

        # Use CUDA events for accurate GPU timing
        start_e = torch.cuda.Event(enable_timing=True)
        end_e   = torch.cuda.Event(enable_timing=True)

        runs = RUNS

        with torch.no_grad():
            start_e.record()
            for _ in range(runs):
                pt_ln(x)
            end_e.record()
            torch.cuda.synchronize()
            pt_ms = start_e.elapsed_time(end_e) / runs

            start_e.record()
            for _ in range(runs):
                fast_ln(x)
            end_e.record()
            torch.cuda.synchronize()
            our_ms = start_e.elapsed_time(end_e) / runs

        _row("LayerNorm CUDA", (N, C), pt_ms, our_ms)


def bench_layer_norm_mps():
    """LayerNorm: custom Metal kernel vs PyTorch MPS vs CPU."""
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("  (no MPS device — skipping Apple Silicon GPU benchmark)")
        return

    import time

    device = "mps"
    print(f"  Device: Apple Silicon GPU (MPS)")

    for N, C in LAYERNORM_SIZES:
        x_cpu = torch.randn(N, C)
        x_mps = x_cpu.to(device)

        # Create CPU and MPS modules SEPARATELY.
        # nn.Module.to() is IN-PLACE: after fast_ln.to("mps"), fast_ln itself
        # is on MPS. Calling fast_ln(x_cpu) would then pass a CPU input to an
        # MPS module — device mismatch in C++ → segfault.
        # Fix: create dedicated CPU and MPS module instances from the start.
        ref_ln = nn.LayerNorm(C)                   # reference weights (CPU)
        fast_ln_cpu = FastLayerNorm(C)              # CPU module stays on CPU
        fast_ln_mps = FastLayerNorm(C).to(device)  # separate MPS module
        pt_ln_mps   = nn.LayerNorm(C).to(device)
        with torch.no_grad():
            fast_ln_cpu.weight.copy_(ref_ln.weight)
            fast_ln_cpu.bias.copy_(ref_ln.bias)
            fast_ln_mps.weight.copy_(ref_ln.weight.to(device))
            fast_ln_mps.bias.copy_(ref_ln.bias.to(device))
            pt_ln_mps.weight.copy_(ref_ln.weight.to(device))
            pt_ln_mps.bias.copy_(ref_ln.bias.to(device))

        # CPU baseline (fast_ln_cpu is strictly on CPU — no device mismatch)
        with torch.no_grad():
            cpu_ms = benchmark(lambda: fast_ln_cpu(x_cpu), runs=RUNS, warmup=WARMUP)

        # MPS timing uses torch.mps.synchronize() for accurate GPU measurement
        def time_mps(fn, runs=RUNS, warmup=WARMUP):
            with torch.no_grad():
                for _ in range(warmup):
                    fn()
                torch.mps.synchronize()
                t0 = time.perf_counter()
                for _ in range(runs):
                    fn()
                torch.mps.synchronize()
                return (time.perf_counter() - t0) * 1000 / runs

        pt_mps_ms   = time_mps(lambda: pt_ln_mps(x_mps))
        fast_mps_ms = time_mps(lambda: fast_ln_mps(x_mps))  # our Metal kernel

        speedup_vs_cpu = cpu_ms / fast_mps_ms
        speedup_vs_pt  = pt_mps_ms / fast_mps_ms
        print(f"  LayerNorm {str((N,C)):<14} | CPU {cpu_ms:.3f}ms | "
              f"PT-MPS {pt_mps_ms:.3f}ms | Ours-MPS {fast_mps_ms:.3f}ms | "
              f"vs PT-MPS {speedup_vs_pt:.2f}x | vs CPU {speedup_vs_cpu:.2f}x")


if __name__ == "__main__":
    print("\ntorchfast vs PyTorch CPU benchmark\n")
    _header()
    bench_fused_linear()
    bench_layer_norm()
    bench_attention()
    bench_focal_loss()
    print()

    import torch
    if torch.backends.mps.is_available():
        print("\ntorchfast LayerNorm: MPS (Apple Silicon GPU) benchmark\n")
        bench_layer_norm_mps()
        print()
    else:
        print("\n(No MPS GPU — Apple Silicon GPU benchmark skipped)")

    import torch
    if torch.cuda.is_available():
        print(f"\ntorchfast vs PyTorch GPU benchmark  [{torch.cuda.get_device_name(0)}]\n")
        _header()
        bench_layer_norm_cuda()
        print()
    else:
        print("\n(No CUDA GPU detected — GPU benchmark skipped)")
        print("To run GPU benchmarks: execute on a machine with an NVIDIA GPU")
