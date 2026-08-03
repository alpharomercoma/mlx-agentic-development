import os, sys, inspect
sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx

import solution


def test_accumulates_correctly():
    step = solution.make_counter()
    total = 0.0
    for i in range(1, 12):
        x = mx.full((4,), float(i), dtype=mx.float32)
        mx.eval(x)
        got = step(x)
        mx.eval(got)
        total += 4.0 * i
        assert abs(got.item() - total) < 1e-3, (
            f"call {i}: got {got.item()}, expected {total}. "
            "A compiled function freezes captured state at trace time; mutating it "
            "afterwards is silently ignored."
        )


def test_counters_are_independent():
    a, b = solution.make_counter(), solution.make_counter()
    x = mx.ones((3,), dtype=mx.float32)
    mx.eval(x)
    for _ in range(4):
        ra = a(x)
    rb = b(x)
    mx.eval(ra, rb)
    assert abs(ra.item() - 12.0) < 1e-3, f"counter a should hold 12, holds {ra.item()}"
    assert abs(rb.item() - 3.0) < 1e-3, f"counter b should hold 3, holds {rb.item()}"


def test_varying_shapes_still_accumulate():
    step = solution.make_counter()
    total = 0.0
    for n in (2, 5, 2, 9, 5):
        x = mx.ones((n,), dtype=mx.float32)
        mx.eval(x)
        got = step(x)
        mx.eval(got)
        total += n
        assert abs(got.item() - total) < 1e-3, f"n={n}: got {got.item()}, expected {total}"


def test_uses_compile():
    src = inspect.getsource(solution)
    assert "compile" in src, "the per-step computation must be wrapped with mx.compile"
