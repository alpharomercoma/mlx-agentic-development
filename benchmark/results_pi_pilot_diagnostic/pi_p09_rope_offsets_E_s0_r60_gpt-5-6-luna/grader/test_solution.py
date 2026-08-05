import os, sys
sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx

import solution

BASE, SCALE = 10000.0, 1.0


def reference(x, offsets):
    # Per-batch scalar-offset calls, which is the unambiguous meaning.
    outs = []
    for b in range(x.shape[0]):
        outs.append(
            mx.fast.rope(
                x[b : b + 1], x.shape[-1], traditional=False, base=BASE,
                scale=SCALE, offset=int(offsets[b].item()),
            )
        )
    return mx.concatenate(outs, axis=0)


def rel_fro(got, exp):
    return (mx.sqrt(mx.sum((got - exp) ** 2)) / mx.sqrt(mx.sum(exp**2))).item()


def test_matches_per_sequence_reference():
    mx.random.seed(0)
    x = mx.random.normal((4, 2, 6, 16)).astype(mx.float32)
    offsets = mx.array([0, 3, 17, 128], dtype=mx.int32)
    mx.eval(x, offsets)
    got = solution.rope_batched(x, offsets)
    exp = reference(x, offsets)
    mx.eval(got, exp)
    assert got.shape == x.shape, f"{got.shape} != {x.shape}"
    assert got.dtype == mx.float32
    err = rel_fro(got, exp)
    assert err < 1e-5, f"relative Frobenius error {err:.2e}"


def test_offsets_actually_used():
    # Identical content in every batch element with DIFFERENT offsets must produce
    # different rows. A solution that ignores offsets passes the shape checks.
    mx.random.seed(1)
    one = mx.random.normal((1, 2, 6, 16)).astype(mx.float32)
    x = mx.concatenate([one, one], axis=0)
    offsets = mx.array([0, 5], dtype=mx.int32)
    mx.eval(x, offsets)
    got = solution.rope_batched(x, offsets)
    mx.eval(got)
    diff = mx.max(mx.abs(got[0] - got[1])).item()
    assert diff > 1e-3, "both batch elements identical; per-sequence offsets ignored"


def test_zero_offsets_match_plain_rope():
    mx.random.seed(2)
    x = mx.random.normal((3, 1, 8, 32)).astype(mx.float32)
    offsets = mx.zeros((3,), dtype=mx.int32)
    mx.eval(x, offsets)
    got = solution.rope_batched(x, offsets)
    exp = mx.fast.rope(x, 32, traditional=False, base=BASE, scale=SCALE, offset=0)
    mx.eval(got, exp)
    assert rel_fro(got, exp) < 1e-5
