"""Fused scale-and-bias operation implemented with a custom Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel creates (and potentially JIT-compiles) a Metal
# library, so keep this at module scope and reuse it for every invocation.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_float32",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source=r"""
        uint index = thread_position_in_grid.x;
        out[index] = x[index] * scale[0] + bias[0];
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU-kernel launch.

    ``scale`` and ``bias`` are scalar GPU inputs rather than values baked into
    the source, so changing either does not create or compile another kernel.
    """
    if x.dtype != mx.float32:
        raise TypeError("x must have dtype mlx.core.float32")

    if x.size == 0:
        # A zero-sized dispatch is invalid, and there is no elementwise work to
        # perform.  This also preserves the requested shape and dtype.
        return mx.zeros(x.shape, dtype=x.dtype)

    scale_array = mx.array(scale, dtype=mx.float32)
    bias_array = mx.array(bias, dtype=mx.float32)
    return _SCALE_BIAS_KERNEL(
        inputs=[x, scale_array, bias_array],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
