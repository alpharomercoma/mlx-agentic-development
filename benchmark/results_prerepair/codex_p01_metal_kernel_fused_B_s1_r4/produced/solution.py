"""Fused scale-and-bias operation implemented with a custom Metal kernel."""

import mlx.core as mx


_THREADS_PER_THREADGROUP = 256

# Constructing a metal_kernel can compile a Metal library, so keep this at module
# scope.  The returned callable accepts Python scalars as inputs; MLX passes those
# as scalar arrays to the generated kernel.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_f32",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        // x may be a view.  Output arrays made by metal_kernel are contiguous.
        uint x_elem = elem_to_loc(elem, x_shape, x_strides, x_ndim);
        out[elem] = x[x_elem] * scale[0] + bias[0];
    """,
    ensure_row_contiguous=False,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU-kernel launch.

    ``x`` must be an MLX float32 array.  ``scale`` and ``bias`` deliberately
    remain Python floats: MLX supplies them to the custom kernel as scalar inputs.
    """
    if x.dtype != mx.float32:
        raise TypeError("fused_scale_bias expects an MLX float32 array")
    if x.size == 0:
        # There is no valid zero-sized Metal dispatch, and no arithmetic to do.
        return x

    # A dispatch may not have a threadgroup dimension larger than its grid.
    threads = min(_THREADS_PER_THREADGROUP, x.size)
    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, float(scale), float(bias)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    return out
