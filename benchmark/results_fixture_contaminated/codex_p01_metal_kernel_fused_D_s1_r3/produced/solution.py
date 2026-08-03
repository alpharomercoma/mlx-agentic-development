"""Fused elementwise scale-and-bias implemented with a custom Metal kernel."""

import mlx.core as mx


# Construct the JIT custom-kernel wrapper once.  MLX caches the compiled Metal
# program, so calls below only enqueue this kernel with their current inputs.
_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i < n) {
        out[i] = inp[i] * scale + bias;
    }
"""

_FUSED_SCALE_BIAS = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["inp", "scale", "bias", "n"],
    output_names=["out"],
    source=_SOURCE,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using the custom Metal kernel above."""
    n = x.size
    (out,) = _FUSED_SCALE_BIAS(
        inputs=[
            x,
            mx.array(scale, dtype=mx.float32),
            mx.array(bias, dtype=mx.float32),
            mx.array(n, dtype=mx.uint32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, max(n, 1)), 1, 1),
    )
    return out
