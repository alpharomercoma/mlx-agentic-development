Create `solution.py` exposing:

    safe_take(x, idx, fill) -> mx.array

`x` is a 1-D float32 MLX array of length N. `idx` is an int32 MLX array of arbitrary
shape containing indices into `x`, which **may be out of range in either direction**
(negative beyond -N, or >= N). `fill` is a Python float.

Return an array with the same shape as `idx` where:

  * for an index i with 0 <= i < N, the result is `x[i]`
  * for an index i with -N <= i < 0, the result is `x[N + i]` (Python-style wrapping)
  * for any other index, the result is `fill`

The result must be float32. The function must be safe for any integer values in
`idx`, including very large ones.
