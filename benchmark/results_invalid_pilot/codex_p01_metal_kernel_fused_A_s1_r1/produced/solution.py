"""A fused scale-and-bias operation implemented with an MLX Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel JIT-compiles the Metal source, so this must remain at
# module scope rather than in fused_scale_bias's hot path.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source=r"""
        uint i = thread_position_in_grid.x;
        if (i < n_elements) {
            out[i] = x[i] * scale + bias;
        }
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using a single custom GPU-kernel launch.

    ``scale`` and ``bias`` are passed as scalar kernel arguments, not converted to
    MLX arrays, so no scalar-array allocation is needed for each invocation.
    """
    n_elements = x.size
    if n_elements == 0:
        # No arithmetic is required for an empty tensor, and Metal cannot dispatch
        # a zero-sized grid.
        return mx.empty(x.shape, dtype=x.dtype)

    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n_elements, 1, 1),
        threadgroup=(min(256, n_elements), 1, 1),
    )
    return out
