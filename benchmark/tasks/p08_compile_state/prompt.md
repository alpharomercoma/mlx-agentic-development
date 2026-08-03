Create `solution.py` exposing:

    make_counter() -> callable

`make_counter()` returns a function `step(x)` that:

  * keeps an internal running total, starting at zero
  * on each call, adds `mx.sum(x)` to that running total
  * returns the running total after the addition, as an `mx.array` scalar

For performance, the per-step computation must be wrapped with `mx.compile`. Each
counter returned by `make_counter()` must be independent of any other.

Correctness is judged by the returned values across many successive calls.
