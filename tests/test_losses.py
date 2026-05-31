import os
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "csrc")
_FLAGS   = ["-O3", "-march=native", "-ffast-math"]

ext = load(
    name="focal_loss_test",
    sources=[os.path.join(_EXT_DIR, "losses.cpp")],
    extra_cflags=_FLAGS,
    verbose=False,
)


def _ref_focal_loss(logits, targets, alpha, gamma):
    probs = torch.softmax(logits, dim=1)
    N, C  = logits.shape
    pt    = probs[torch.arange(N), targets].clamp(min=1e-7)
    fl    = -alpha * (1 - pt) ** gamma * torch.log(pt)
    return fl.mean()


@pytest.mark.parametrize("N,C", [(1, 4), (32, 10), (128, 100)])
@pytest.mark.parametrize("alpha,gamma", [(0.25, 2.0), (0.5, 1.0), (1.0, 0.0)])
def test_forward_correctness(N, C, alpha, gamma):
    logits  = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))

    got = ext.forward(logits, targets, alpha, gamma)
    ref = _ref_focal_loss(logits, targets, alpha, gamma)

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N,C", [(16, 8), (64, 20)])
def test_backward_correctness(N, C):
    logits  = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    alpha, gamma = 0.25, 2.0

    # Custom kernel backward
    grad_out = torch.ones(())
    gl = ext.backward(grad_out, logits, targets, alpha, gamma)

    # Reference via autograd
    lr = logits.clone().requires_grad_(True)
    _ref_focal_loss(lr, targets, alpha, gamma).backward()

    torch.testing.assert_close(gl, lr.grad, rtol=1e-4, atol=1e-4)


def test_gradcheck():
    N, C = 4, 5
    logits  = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, C, (N,))

    def fn(l):
        return _ref_focal_loss(l, targets, 0.25, 2.0)

    assert torch.autograd.gradcheck(fn, (logits,), eps=1e-6, atol=1e-4)


def test_batch_size_one():
    logits  = torch.randn(1, 10)
    targets = torch.tensor([3])
    loss = ext.forward(logits, targets, 0.25, 2.0)
    assert loss.shape == ()


def test_large_tensor():
    logits  = torch.randn(1024, 1000)
    targets = torch.randint(0, 1000, (1024,))
    loss = ext.forward(logits, targets, 0.25, 2.0)
    assert loss.shape == ()


def test_wrong_logits_dim_raises():
    logits  = torch.randn(4)  # 1D, should be 2D
    targets = torch.randint(0, 4, (4,))
    with pytest.raises(RuntimeError):
        ext.forward(logits, targets, 0.25, 2.0)


def test_batch_mismatch_raises():
    logits  = torch.randn(8, 10)
    targets = torch.randint(0, 10, (4,))  # wrong batch
    with pytest.raises(RuntimeError):
        ext.forward(logits, targets, 0.25, 2.0)
