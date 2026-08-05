import os, sys, inspect
sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx
import pytest

import solution


@pytest.mark.parametrize("shape", [(4, 8), (16, 256), (3, 1000), (7, 33), (64, 1), (2, 4097)])
def test_matches_reference(shape):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.float32)
    mx.eval(x)
    got = solution.row_sum(x)
    exp = mx.sum(x, axis=1)
    mx.eval(got, exp)
    assert got.shape == (shape[0],), f"{got.shape} != {(shape[0],)}"
    assert got.dtype == mx.float32
    err = mx.max(mx.abs(got - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-6
    assert err / scale < 1e-4, f"max rel err {err / scale:.2e} for shape {shape}"


def test_repeated_calls_are_stable():
    # An accumulating kernel whose output buffer is never initialised keeps
    # whatever was in memory, so results drift between identical calls.
    mx.random.seed(1)
    x = mx.random.normal((32, 512)).astype(mx.float32)
    mx.eval(x)
    first = solution.row_sum(x)
    mx.eval(first)
    for _ in range(5):
        again = solution.row_sum(x)
        mx.eval(again)
        assert mx.allclose(first, again, atol=1e-5).item(), "results differ between identical calls"


def test_zeros_give_zeros():
    # The sharpest test for an uninitialised output: the true answer is exactly 0,
    # so any garbage left in the buffer shows up immediately.
    x = mx.zeros((16, 300), dtype=mx.float32)
    mx.eval(x)
    got = solution.row_sum(x)
    mx.eval(got)
    assert mx.max(mx.abs(got)).item() < 1e-6, f"expected zeros, got max {mx.max(mx.abs(got)).item():.3e}"


def test_uses_a_custom_kernel():
    src = inspect.getsource(solution)
    assert "metal_kernel" in src, "must use MLX's custom kernel facility"
