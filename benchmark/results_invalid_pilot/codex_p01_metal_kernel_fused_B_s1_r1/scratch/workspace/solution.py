"""A fused scale-and-bias operation implemented with a custom Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel JIT-compiles its Metal library, so keep the compiled
# kernel at module scope rather than rebuilding it for every invocation.
_FUSED_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < n_elements) {
            out[i] = x[i] * scale + bias;
        }
    """,
)


def fused_scale_bias(x, scale, bias):
    """Return ``x * scale + bias`` using the cached custom GPU kernel.

    ``scale`` and ``bias`` are deliberately passed as Python scalars, which MLX
    binds to the Metal kernel by value.
    """
    n_elements = x.size
    # A one-thread launch makes an empty array safe while the bounds check ensures
    # it does not access its zero-sized buffers.
    grid_size = max(n_elements, 1)
    threadgroup_size = min(grid_size, 256)
    (out,) = _FUSED_SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(grid_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
    )
    return out
