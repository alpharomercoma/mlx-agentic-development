"""Fused scale-and-bias operation implemented as an MLX Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel JIT-compiles its Metal library, so this deliberately
# lives at module scope instead of in fused_scale_bias's hot path.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_f32",
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
    """Return ``x * scale + bias`` using one custom GPU elementwise kernel."""
    # Keep scalar bindings float32, matching the required input/output dtype and
    # avoiding an unintended promotion in the generated Metal function.
    scale = float(scale)
    bias = float(bias)
    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[x, scale, bias],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        # A nonzero grid also makes empty shapes safe; the bounds check performs
        # no memory access in that case.
        grid=(max(x.size, 1), 1, 1),
        threadgroup=(min(max(x.size, 1), 256), 1, 1),
    )
    return out
