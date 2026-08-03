"""Fused float32 scale-and-bias operation implemented as a Metal kernel."""

import mlx.core as mx


# Keep this at module scope: constructing a metal_kernel creates a Metal library
# and must not be part of the repeated-call path.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_float32",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < n_elements) {
            out[i] = x[i] * scale + bias;
        }
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU elementwise kernel."""
    n_elements = x.size
    if n_elements == 0:
        # There are no elementwise operations to dispatch for an empty tensor.
        return mx.empty(x.shape, dtype=x.dtype)

    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n_elements, 1, 1),
        threadgroup=(min(256, n_elements), 1, 1),
    )
    return out
