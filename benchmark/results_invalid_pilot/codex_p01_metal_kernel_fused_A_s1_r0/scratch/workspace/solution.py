"""Fused scale-and-bias operation implemented with an MLX Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel JIT-compiles its Metal library, so this is deliberately
# module-scoped rather than part of fused_scale_bias's hot path.
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
    """Return ``x * scale + bias`` using the cached custom GPU kernel.

    ``x`` is expected to be a float32 MLX array; ``scale`` and ``bias`` are bound
    to the kernel as by-value scalar arguments.
    """
    # There is no elementwise work for an empty tensor, and Metal dispatch requires
    # a nonzero grid dimension.  Returning x preserves its shape and float32 dtype.
    if x.size == 0:
        return x

    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out
