import os, sys
sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx
import pytest

import solution

MODE = "mxfp4"
# Measured on this machine (MLX 0.32.0, M5), same method as p04:
#   correct mxfp4 quantized_matmul vs dequantised reference   4.1e-4
#   full-precision matmul          vs dequantised reference   1.2e-1  (290x larger)
# 5e-3 sits 12x above a correct implementation and 24x below one that skipped
# quantisation. mxfp4 is a coarser format than 4-bit affine, so the gap between a
# correct result and an unquantised one is wider here than in p04.
QMM_TOL = 5e-3
MIN_QUANT_NOISE = 5e-2


def rel_fro(got, exp):
    return (mx.sqrt(mx.sum((got - exp) ** 2)) / mx.sqrt(mx.sum(exp**2))).item()


def _reference(w, x):
    packed = mx.quantize(w, mode=MODE)
    wq, scales = packed[0], packed[1]
    w_deq = mx.dequantize(wq, scales, mode=MODE, dtype=mx.float32)
    return x @ w_deq.T


@pytest.mark.parametrize("shape", [(128, 256), (64, 128), (256, 512)])
def test_matches_dequantised_reference(shape):
    out_f, in_f = shape
    mx.random.seed(0)
    w = mx.random.normal((out_f, in_f)).astype(mx.float32)
    x = mx.random.normal((8, in_f)).astype(mx.float32)
    mx.eval(w, x)
    got = solution.matmul_mxfp4(x, solution.pack_mxfp4(w))
    exp = _reference(w, x)
    mx.eval(got, exp)
    assert got.shape == exp.shape
    assert got.dtype == mx.float32
    err = rel_fro(got, exp)
    assert err < QMM_TOL, f"relative Frobenius error {err:.2e}"


def test_actually_quantised():
    mx.random.seed(2)
    w = mx.random.normal((128, 256)).astype(mx.float32)
    x = mx.random.normal((8, 256)).astype(mx.float32)
    mx.eval(w, x)
    got = solution.matmul_mxfp4(x, solution.pack_mxfp4(w))
    full = x @ w.T
    mx.eval(got, full)
    err = rel_fro(got, full)
    assert err > MIN_QUANT_NOISE, f"too close to full precision ({err:.2e}); not quantised"
