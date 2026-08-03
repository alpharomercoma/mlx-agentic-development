"""Fused scale-and-bias operation implemented with an MLX Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel compiles/caches the kernel definition.  Keep it at
# module scope so calls to fused_scale_bias only need to dispatch it.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_f32",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source=r"""
        uint i = thread_position_in_grid.x;
        out[i] = x[i] * scale[0] + bias[0];
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU-kernel dispatch.

    ``x`` is flattened only for the kernel's indexing; its original shape is
    supplied when MLX creates the output.
    """
    return _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
