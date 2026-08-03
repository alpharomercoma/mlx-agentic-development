import inspect
import os
import sys

sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx
import pytest
import solution


def ref(x, s, b):
    return x * s + b


@pytest.mark.parametrize("shape", [(8,), (1024,), (33,), (7, 13), (2, 3, 5), (256, 256)])
def test_matches_reference(shape):
    x = mx.random.normal(shape).astype(mx.float32)
    mx.eval(x)
    got = solution.fused_scale_bias(x, 2.5, -1.25)
    mx.eval(got)
    assert got.shape == tuple(shape)
    assert got.dtype == mx.float32
    assert mx.allclose(got, ref(x, 2.5, -1.25), atol=1e-5).item()


def test_non_multiple_of_threadgroup():
    # Sizes that are not a multiple of a typical threadgroup width catch kernels
    # that dispatch threadblocks instead of threads, or that skip a tail.
    for n in (1, 3, 129, 257, 1023):
        x = mx.random.normal((n,)).astype(mx.float32)
        mx.eval(x)
        got = solution.fused_scale_bias(x, -0.5, 3.0)
        mx.eval(got)
        assert mx.allclose(got, ref(x, -0.5, 3.0), atol=1e-5).item(), f"failed at n={n}"


def test_no_uninitialised_output():
    # An output buffer that is never fully written keeps whatever was in memory.
    # Running twice with different inputs surfaces it.
    a = mx.zeros((4096,), dtype=mx.float32)
    b = mx.ones((4096,), dtype=mx.float32) * 9.0
    mx.eval(a, b)
    ra = solution.fused_scale_bias(a, 1.0, 0.0)
    mx.eval(ra)
    rb = solution.fused_scale_bias(b, 1.0, 0.0)
    mx.eval(rb)
    assert mx.allclose(ra, a, atol=1e-6).item()
    assert mx.allclose(rb, b, atol=1e-6).item()


def test_uses_a_custom_kernel():
    src = inspect.getsource(solution)
    assert "metal_kernel" in src, "must use MLX's custom kernel facility"
