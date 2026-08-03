"""A fused scale-and-bias operation implemented with an MLX Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel creates a Metal library (and can JIT compile it), so
# keep it at module scope rather than rebuilding it for every invocation.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_f32",
    input_names=["x", "scale", "bias", "n_elements"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < static_cast<uint>(n_elements)) {
            out[i] = x[i] * scale + bias;
        }
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using a custom GPU kernel.

    ``x`` is expected to be a float32 MLX array.  MLX makes non-contiguous input
    row-contiguous for this kernel, preserving the elementwise semantics.
    """
    # A zero-sized dispatch is invalid; the expression has no elements to compute,
    # so the input itself is already the correctly shaped and typed empty result.
    if x.size == 0:
        return x

    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias, x.size],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(min(256, x.size), 1, 1),
    )
    return out
