import os, sys

sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx
import pytest

import solution

BITS, GROUP = 4, 64

# Tolerances set from measurement on this machine (MLX 0.32.0, M5). mx.quantized_matmul
# does not equal dequantise-then-matmul bit for bit; it accumulates differently:
#     correct quantized_matmul  vs dequantised reference   7.7e-4
#     full-precision matmul     vs dequantised reference   9.3e-2   (120x larger)
# 5e-3 sits 6.5x above a correct implementation and 18x below one that skipped
# quantisation entirely.
QMM_TOL = 5e-3
# A genuinely quantised result must differ from the full-precision product by
# roughly the quantisation noise, measured at 9.3e-2.
MIN_QUANT_NOISE = 1e-2


def rel_fro(got, exp):
    return (mx.sqrt(mx.sum((got - exp) ** 2)) / mx.sqrt(mx.sum(exp**2))).item()


@pytest.mark.parametrize("shape", [(128, 256), (64, 128), (256, 512)])
def test_matches_dequantised_reference(shape):
    out_f, in_f = shape
    mx.random.seed(0)
    w = mx.random.normal((out_f, in_f)).astype(mx.float32)
    x = mx.random.normal((8, in_f)).astype(mx.float32)
    mx.eval(w, x)

    packed = solution.pack(w)
    got = solution.qmatmul(x, packed)
    mx.eval(got)

    # Reference: quantise with the same settings, dequantise, ordinary matmul.
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
    w_deq = mx.dequantize(
        wq, scales, biases, group_size=GROUP, bits=BITS, dtype=mx.float32
    )
    exp = x @ w_deq.T
    mx.eval(exp)

    assert got.shape == exp.shape, f"{got.shape} != {exp.shape}"
    assert got.dtype == mx.float32
    err = rel_fro(got, exp)
    assert err < QMM_TOL, f"relative Frobenius error {err:.2e} vs dequantised reference"


def test_batched_input():
    mx.random.seed(1)
    w = mx.random.normal((96, 128)).astype(mx.float32)
    x = mx.random.normal((3, 5, 128)).astype(mx.float32)
    mx.eval(w, x)
    got = solution.qmatmul(x, solution.pack(w))
    mx.eval(got)
    assert got.shape == (3, 5, 96)


def test_actually_quantised():
    # A solution that ignores quantisation and does a plain matmul would match the
    # float reference far too well. Compare against the FULL-precision product: a
    # genuinely quantised result must differ from it measurably.
    mx.random.seed(2)
    w = mx.random.normal((128, 256)).astype(mx.float32)
    x = mx.random.normal((8, 256)).astype(mx.float32)
    mx.eval(w, x)
    got = solution.qmatmul(x, solution.pack(w))
    full = x @ w.T
    mx.eval(got, full)
    err = rel_fro(got, full)
    assert err > MIN_QUANT_NOISE, (
        f"result is too close to the full-precision product ({err:.2e}); "
        "the weights do not appear to be quantised"
    )
