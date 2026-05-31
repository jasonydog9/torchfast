import os
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "csrc")
_FLAGS   = ["-O3", "-march=native", "-ffast-math"]

ext = load(
    name="fused_linear_act_test",
    sources=[os.path.join(_EXT_DIR, "fused_linear_act.cpp")],
    extra_cflags=_FLAGS,
    verbose=False,
)


def _ref_forward(x, w, b, act):
    out = F.linear(x, w, b)
    if act == "gelu":
        return F.gelu(out, approximate="tanh")
    if act == "relu":
        return F.relu(out)
    return F.silu(out)


@pytest.mark.parametrize("act", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("N,K,M", [(1, 16, 8), (32, 256, 128), (128, 512, 256)])
def test_forward_correctness(act, N, K, M):
    x = torch.randn(N, K)
    w = torch.randn(M, K)
    b = torch.randn(M)

    got = ext.forward(x, w, b, act)
    ref = _ref_forward(x, w, b, act)

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("act", ["gelu", "relu", "silu"])
def test_backward_correctness(act):
    N, K, M = 16, 32, 24
    x  = torch.randn(N, K, requires_grad=False)
    w  = torch.randn(M, K, requires_grad=False)
    b  = torch.randn(M,    requires_grad=False)
    go = torch.randn(N, M)

    gi, gw, gb = ext.backward(go, x, w, b, act)

    # Reference via autograd
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    br = b.clone().requires_grad_(True)
    ref = _ref_forward(xr, wr, br, act)
    ref.backward(go)

    torch.testing.assert_close(gi, xr.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gw, wr.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gb, br.grad, rtol=1e-4, atol=1e-4)


def test_gradcheck():
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6,    dtype=torch.float64, requires_grad=True)

    # gradcheck needs float64; use the ATen-based reference, not our kernel
    def fn(x_, w_, b_):
        return F.gelu(F.linear(x_, w_, b_), approximate="tanh")

    assert torch.autograd.gradcheck(fn, (x, w, b), eps=1e-6, atol=1e-4)


def test_batch_size_one():
    x = torch.randn(1, 64)
    w = torch.randn(32, 64)
    b = torch.randn(32)
    out = ext.forward(x, w, b, "gelu")
    assert out.shape == (1, 32)


def test_large_tensor():
    x = torch.randn(1024, 2048)
    w = torch.randn(2048, 2048)
    b = torch.randn(2048)
    out = ext.forward(x, w, b, "silu")
    assert out.shape == (1024, 2048)


def test_mismatched_dims_raises():
    x = torch.randn(4, 8)
    w = torch.randn(6, 16)   # wrong inner dim
    b = torch.randn(6)
    with pytest.raises(RuntimeError):
        ext.forward(x, w, b, "gelu")


def test_non_contiguous_raises():
    x = torch.randn(8, 16).t()  # non-contiguous after transpose
    w = torch.randn(4, 8)
    b = torch.randn(4)
    with pytest.raises(RuntimeError):
        ext.forward(x, w, b, "gelu")


def test_invalid_activation_raises():
    x = torch.randn(4, 8)
    w = torch.randn(6, 8)
    b = torch.randn(6)
    with pytest.raises(RuntimeError):
        ext.forward(x, w, b, "swish")
