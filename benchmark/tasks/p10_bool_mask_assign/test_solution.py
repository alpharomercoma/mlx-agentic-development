import os, sys
sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx

import solution

FILL = -99.0


def expected(xs, ids, n, fill):
    out = []
    for i in ids:
        if 0 <= i < n:
            out.append(xs[i])
        elif -n <= i < 0:
            out.append(xs[n + i])
        else:
            out.append(fill)
    return out


def test_in_range():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ids = [0, 2, 4, 1]
    got = solution.safe_take(mx.array(xs, dtype=mx.float32), mx.array(ids, dtype=mx.int32), FILL)
    mx.eval(got)
    assert got.tolist() == expected(xs, ids, len(xs), FILL)


def test_negative_wrapping():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ids = [-1, -5, -3]
    got = solution.safe_take(mx.array(xs, dtype=mx.float32), mx.array(ids, dtype=mx.int32), FILL)
    mx.eval(got)
    assert got.tolist() == expected(xs, ids, len(xs), FILL)


def test_out_of_range_is_filled_not_garbage():
    # MLX does not bounds-check. An unguarded gather reads whatever is in memory,
    # which is usually not the fill value and is not reproducible.
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ids = [5, 99, -6, -100000, 2, 1000000]
    got = solution.safe_take(mx.array(xs, dtype=mx.float32), mx.array(ids, dtype=mx.int32), FILL)
    mx.eval(got)
    assert got.tolist() == expected(xs, ids, len(xs), FILL), (
        "out-of-range indices did not produce the fill value; MLX indexing is "
        "unchecked, so they must be masked explicitly"
    )


def test_shape_and_dtype_preserved():
    xs = mx.arange(10).astype(mx.float32)
    ids = mx.array([[0, 20], [-1, -50]], dtype=mx.int32)
    mx.eval(xs, ids)
    got = solution.safe_take(xs, ids, FILL)
    mx.eval(got)
    assert got.shape == (2, 2), got.shape
    assert got.dtype == mx.float32
    assert got.tolist() == [[0.0, FILL], [9.0, FILL]]


def test_repeatable():
    xs = mx.arange(8).astype(mx.float32)
    ids = mx.array([100, -100, 3], dtype=mx.int32)
    mx.eval(xs, ids)
    runs = []
    for _ in range(5):
        r = solution.safe_take(xs, ids, FILL)
        mx.eval(r)
        runs.append(r.tolist())
    assert all(r == runs[0] for r in runs), f"results not reproducible: {runs}"
