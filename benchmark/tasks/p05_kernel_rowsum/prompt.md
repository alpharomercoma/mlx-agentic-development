Create `solution.py` exposing:

    row_sum(x) -> mx.array

`x` is a 2-D MLX float32 array of shape [M, N]. Return a 1-D float32 array of shape
[M] where element i is the sum of row i of `x`.

The reduction must be performed by a custom GPU kernel you write with MLX's
custom-kernel facility, using more than one thread per row so that the work within a
row is shared. Do not implement it with MLX's built-in reduction operators.

N is not guaranteed to be a multiple of any particular number.
