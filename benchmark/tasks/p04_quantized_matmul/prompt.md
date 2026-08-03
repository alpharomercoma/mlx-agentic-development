Create `solution.py` exposing two functions:

    pack(w) -> object
    qmatmul(x, packed) -> mx.array

`w` is a float32 weight matrix of shape [out_features, in_features]. `pack` must
quantise it to 4 bits with a group size of 64 using MLX's built-in quantisation, and
return whatever object `qmatmul` needs.

`qmatmul(x, packed)` takes `x` of shape [..., in_features] and must return
`x @ w.T` computed with MLX's quantised matrix-multiply, giving shape
[..., out_features] and dtype float32.

The result will be compared against dequantising the packed weights and doing an
ordinary matmul, so it must agree with that to tight tolerance.
