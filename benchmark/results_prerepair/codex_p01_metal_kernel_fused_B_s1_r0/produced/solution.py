"""A fused scale-and-bias operation implemented as a custom Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel creates a Metal library (and can JIT compile it), so
# keep it at module scope and reuse it for every invocation.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_f32",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        out[elem] = x[elem] * scale[0] + bias[0];
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU-kernel launch.

    ``x`` is specified to be float32.  The scalar arrays are runtime kernel
    inputs so their Python values may change between calls without rebuilding
    the Metal kernel.
    """
    if x.size == 0:
        # A zero-thread Metal launch is invalid.  No arithmetic is needed when
        # there are no elements, so only allocate the correctly shaped result.
        return mx.empty(x.shape, dtype=x.dtype)

    scale_array = mx.array(scale, dtype=mx.float32)
    bias_array = mx.array(bias, dtype=mx.float32)
    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale_array, bias_array],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out
