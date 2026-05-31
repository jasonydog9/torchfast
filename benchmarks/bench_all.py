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


def bench_layer_norm():
    for N, C in SIZES:
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


if __name__ == "__main__":
    print("\ntorchfast vs PyTorch CPU benchmark\n")
    _header()
    bench_fused_linear()
    bench_layer_norm()
    bench_attention()
    bench_focal_loss()
    print()
