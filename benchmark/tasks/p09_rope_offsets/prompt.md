Create `solution.py` exposing:

    rope_batched(x, offsets) -> mx.array

`x` is a float32 array of shape [B, H, T, D] and `offsets` is an int32 array of shape
[B] giving the starting position of each batch element's sequence.

Apply MLX's fast rotary positional embedding to `x`, using each batch element's own
starting offset, with `traditional=False`, `base=10000.0` and `scale=1.0`, rotating
all `D` dimensions. Return an array of the same shape and dtype.

Prefer a single fused call over a Python loop if MLX supports it.
