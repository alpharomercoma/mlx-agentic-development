"""A fused float32 scale-and-bias operation implemented as a Metal kernel."""

import mlx.core as mx


# `metal_kernel` JIT-compiles the source, so keep this at module scope rather than
# constructing it for every invocation of ``fused_scale_bias``.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source=r"""
        uint index = thread_position_in_grid.x;
        if (index < n_elements) {
            out[index] = x[index] * scale + bias;
        }
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU kernel launch.

    ``x`` is expected to be a float32 MLX array.  Python scalar inputs are bound
    to the Metal kernel by value, avoiding scalar-array allocation on each call.
    """
    if x.size == 0:
        # No dispatch is needed for an empty output, and x already has the required
        # shape and dtype.
        return x

    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(min(256, x.size), 1, 1),
    )
    return out
