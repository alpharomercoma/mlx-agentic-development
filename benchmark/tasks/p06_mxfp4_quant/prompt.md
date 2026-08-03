Create `solution.py` exposing:

    pack_mxfp4(w) -> object
    matmul_mxfp4(x, packed) -> mx.array

`w` is a float32 weight matrix of shape [out_features, in_features]. `pack_mxfp4`
must quantise it using MLX's **mxfp4** quantisation mode, at that mode's own default
group size and bit width, and return whatever `matmul_mxfp4` needs.

`matmul_mxfp4(x, packed)` takes `x` of shape [..., in_features] and must return
`x @ w.T` computed with MLX's quantised matrix-multiply in the same mxfp4 mode,
returning shape [..., out_features] and dtype float32.

The result will be compared against dequantising the packed weights in the same mode
and performing an ordinary matmul.
