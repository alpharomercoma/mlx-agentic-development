In this directory, create `solution.py` exposing a function:

    fused_scale_bias(x, scale, bias) -> mx.array

It must compute `x * scale + bias` elementwise, where `x` is an MLX float32 array of
arbitrary shape and `scale` and `bias` are Python floats. The result must have the
same shape and dtype as `x`.

Requirement: the elementwise arithmetic must be performed by a custom GPU kernel that
you write with MLX's custom-kernel facility. Do not implement it with MLX's built-in
array operators.

The function will be called many times in a loop, so avoid per-call setup that could
be done once.
