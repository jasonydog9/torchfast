import os
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "csrc")
_FLAGS   = ["-O3", "-march=native", "-ffast-math"]

ext = load(
    name="fast_normalization_test",
    sources=[os.path.join(_EXT_DIR, "normalization.cpp")],
    extra_cflags=_FLAGS,
    verbose=False,
)


@pytest.mark.parametrize("N,C", [(1, 16), (32, 128), (256, 512)])
def test_forward_correctness(N, C):
    x = torch.randn(N, C)
    w = torch.randn(C)
    b = torch.randn(C)
    eps = 1e-5

    out, _, _ = ext.forward(x, w, b, eps)
    ref = F.layer_norm(x, [C], w, b, eps)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N,C", [(8, 32), (64, 256)])
def test_backward_correctness(N, C):
    x  = torch.randn(N, C)
    w  = torch.randn(C)
    b  = torch.randn(C)
    go = torch.randn(N, C)
    eps = 1e-5

    _, mean, rstd = ext.forward(x, w, b, eps)
    gi, gw, gb   = ext.backward(go, x, w, mean, rstd, eps)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    br = b.clone().requires_grad_(True)
    F.layer_norm(xr, [C], wr, br, eps).backward(go)

    torch.testing.assert_close(gi, xr.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gw, wr.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(gb, br.grad, rtol=1e-4, atol=1e-4)


def test_gradcheck():
    N, C = 4, 8
    x = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    w = torch.randn(C,    dtype=torch.float64, requires_grad=True)
    b = torch.randn(C,    dtype=torch.float64, requires_grad=True)

    def fn(x_, w_, b_):
        return F.layer_norm(x_, [C], w_, b_, 1e-5)

    assert torch.autograd.gradcheck(fn, (x, w, b), eps=1e-6, atol=1e-4)


def test_batch_size_one():
    x = torch.randn(1, 64)
    w = torch.ones(64)
    b = torch.zeros(64)
    out, _, _ = ext.forward(x, w, b, 1e-5)
    assert out.shape == (1, 64)


def test_large_tensor():
    x = torch.randn(1024, 2048)
    w = torch.ones(2048)
    b = torch.zeros(2048)
    out, _, _ = ext.forward(x, w, b, 1e-5)
    assert out.shape == (1024, 2048)


def test_mismatched_weight_raises():
    x = torch.randn(4, 16)
    w = torch.randn(8)   # wrong
    b = torch.randn(16)
    with pytest.raises(RuntimeError):
        ext.forward(x, w, b, 1e-5)


def test_non_2d_input_raises():
    x = torch.randn(4, 8, 16)
    w = torch.randn(16)
    b = torch.randn(16)
    with pytest.raises(RuntimeError):
        ext.forward(x, w, b, 1e-5)
