import math
import os
import sys
import time

sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx
import pytest
import solution


# The reference lives here, not in the workspace, so the agent cannot weaken it.
def reference(q, k, v):
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    t, s = q.shape[2], k.shape[2]
    row = mx.arange(t).reshape(t, 1) + (s - t)
    col = mx.arange(s).reshape(1, s)
    mask = col > row
    scores = mx.where(mask, mx.array(-float("inf"), dtype=scores.dtype), scores)
    w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    return w @ v


SHAPES = [(1, 4, 64, 32), (2, 8, 128, 64), (1, 8, 256, 64), (1, 2, 7, 16)]

# Tolerance is relative Frobenius error, chosen from measurement rather than taste.
# Measured on this machine (MLX 0.32.0, M5):
#     correct float32 via mx.fast.scaled_dot_product_attention   3.7e-4
#     same but computing in bfloat16                             3.9e-3
#     causal mask omitted                                        8.6e-1
#     wrong scale factor                                         1.7
# A fused attention kernel legitimately accumulates in a different order from the
# naive path, so it cannot match bit for bit. 1e-3 sits 2.7x above the correct
# implementation and 3.9x below a bfloat16 shortcut, which the prompt forbids
# because it buys speed by changing the numerics.
REL_FRO_TOL = 1e-3


def rel_frobenius(got, exp):
    return (mx.sqrt(mx.sum((got - exp) ** 2)) / mx.sqrt(mx.sum(exp**2))).item()


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(shape):
    b, h, t, d = shape
    mx.random.seed(0)
    q = mx.random.normal((b, h, t, d)).astype(mx.float32)
    k = mx.random.normal((b, h, t, d)).astype(mx.float32)
    v = mx.random.normal((b, h, t, d)).astype(mx.float32)
    mx.eval(q, k, v)
    got = solution.attention(q, k, v)
    exp = reference(q, k, v)
    mx.eval(got, exp)
    assert got.shape == exp.shape
    assert got.dtype == mx.float32, f"returned {got.dtype}, not float32"
    err = rel_frobenius(got, exp)
    assert err < REL_FRO_TOL, (
        f"relative Frobenius error {err:.2e} exceeds {REL_FRO_TOL:.0e}"
    )


def test_cross_attention_shapes():
    # S != T exercises the lower-right causal alignment.
    mx.random.seed(3)
    q = mx.random.normal((1, 4, 16, 32)).astype(mx.float32)
    k = mx.random.normal((1, 4, 64, 32)).astype(mx.float32)
    v = mx.random.normal((1, 4, 64, 32)).astype(mx.float32)
    mx.eval(q, k, v)
    got, exp = solution.attention(q, k, v), reference(q, k, v)
    mx.eval(got, exp)
    err = rel_frobenius(got, exp)
    assert err < REL_FRO_TOL, (
        f"relative Frobenius error {err:.2e} exceeds {REL_FRO_TOL:.0e}"
    )


def _bench(fn, q, k, v, n=50):
    for _ in range(5):
        mx.eval(fn(q, k, v))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn(q, k, v))
    mx.synchronize()
    return (time.perf_counter() - t0) / n


def test_speedup_reported():
    """Emit the speedup for fast_p. Interleaved to limit thermal drift on a
    fanless chassis; never asserts a speedup, only records one."""
    mx.random.seed(0)
    q = mx.random.normal((2, 8, 512, 64)).astype(mx.float32)
    k = mx.random.normal((2, 8, 512, 64)).astype(mx.float32)
    v = mx.random.normal((2, 8, 512, 64)).astype(mx.float32)
    mx.eval(q, k, v)
    ref_t, sol_t = [], []
    for _ in range(3):
        ref_t.append(_bench(reference, q, k, v))
        sol_t.append(_bench(solution.attention, q, k, v))
    r, s = min(ref_t), min(sol_t)
    print(f"\nSPEEDUP_METRIC baseline_s={r:.6f} solution_s={s:.6f} speedup={r / s:.3f}")
