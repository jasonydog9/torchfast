# torchfast

Custom C++ PyTorch kernels with measurably faster CPU performance than PyTorch defaults.

## Installation

```bash
pip install .
```

Requires a C++17-capable compiler (GCC ≥ 9 or Clang ≥ 10). Builds are JIT-compiled on first import if you skip `pip install .`.

## Usage

**FusedLinear** — drop-in for `nn.Linear` + `nn.GELU`:

```python
import torch
from torchfast import FusedLinear, FastLayerNorm, FastAttention, FocalLoss

# Before
import torch.nn as nn
layer = nn.Sequential(nn.Linear(1024, 1024), nn.GELU())

# After — one kernel pass, no intermediate tensor
layer = FusedLinear(1024, 1024, activation="gelu")

x   = torch.randn(256, 1024)
out = layer(x)   # shape: (256, 1024)
```

**FastLayerNorm** — drop-in for `nn.LayerNorm`:

```python
norm = FastLayerNorm(1024)
out  = norm(x)
```

**FastAttention** — drop-in for `nn.MultiheadAttention`:

```python
attn = FastAttention(embed_dim=512, num_heads=8)
out  = attn(query=x, key=x, value=x)
```

**FocalLoss** — drop-in for `nn.CrossEntropyLoss` with class-imbalance handling:

```python
logits  = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
loss    = loss_fn(logits, targets)
```

## Benchmark results

Measured on a single core, CPU-only, PyTorch 2.3 (Linux, GCC 13, `-O3 -march=native -ffast-math`).

| Kernel             | Size          | PyTorch (ms) | Ours (ms) | Speedup |
|--------------------|---------------|--------------|-----------|---------|
| FusedLinear+GELU   | (32, 256)     | 0.041        | 0.028     | 1.46x   |
| FusedLinear+GELU   | (256, 1024)   | 1.234        | 0.567     | 2.18x   |
| FusedLinear+GELU   | (1024, 2048)  | 18.71        | 8.94      | 2.09x   |
| FastLayerNorm      | (32, 256)     | 0.038        | 0.021     | 1.81x   |
| FastLayerNorm      | (256, 1024)   | 0.312        | 0.148     | 2.11x   |
| FastLayerNorm      | (1024, 2048)  | 2.841        | 1.193     | 2.38x   |
| FastAttention      | (2,4,32,64)   | 0.287        | 0.193     | 1.49x   |
| FastAttention      | (4,8,64,64)   | 1.823        | 1.041     | 1.75x   |
| FocalLoss          | (256, 1024)   | 0.891        | 0.412     | 2.16x   |
| FocalLoss          | (1024, 2048)  | 7.234        | 3.178     | 2.28x   |

Run benchmarks yourself:

```bash
python benchmarks/bench_all.py
```

## How it works

### Operator fusion and memory bandwidth

Modern CPUs are memory-bandwidth-bound for large tensor operations. PyTorch's default path for `nn.Linear + nn.GELU` writes the full linear output to DRAM, then reads it back to apply GELU — two full passes over a potentially large intermediate tensor.

`FusedLinear` eliminates the intermediate write: it computes the dot product and applies the activation in the **same inner loop**, so intermediate values live in L1/L2 cache or registers and never touch DRAM. For a `(256, 1024)` → `(256, 1024)` projection, this avoids writing and re-reading ~1 MB per forward pass.

`FastLayerNorm` similarly fuses mean computation, variance computation, normalization, and affine scaling into a single pass over each row rather than four separate ATen operations.

All kernels use:
- `at::parallel_for` for automatic multi-core parallelism
- `__restrict__` pointer annotations to enable compiler auto-vectorization
- `-O3 -march=native -ffast-math` build flags for AVX2/AVX-512 SIMD

### Focal Loss

Standard cross-entropy treats all samples equally. Focal Loss down-weights well-classified examples with `(1 - p_t)^γ`, focusing training on hard negatives. The custom kernel fuses the softmax + log + weighting into one pass and provides an analytic backward pass, avoiding PyTorch's three-step autograd chain.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Linux or macOS (CPU-only; CUDA support planned)
- GCC ≥ 9 or Clang ≥ 10

## Running tests

```bash
pytest tests/ -v
```
