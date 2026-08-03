"""Fused scale-and-bias operation implemented as a Metal custom kernel."""

import mlx.core as mx


# Constructing a metal_kernel creates its compiled Metal library, so keep it at
# module scope and reuse it for every invocation.
_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["x", "scale", "bias", "n"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < n) {
            out[i] = x[i] * scale + bias;
        }
    """,
)

_THREADGROUP_SIZE = 256


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom Metal kernel launch."""
    n = x.size
    (out,) = _KERNEL(
        inputs=[
            x,
            mx.array(scale, dtype=mx.float32),
            mx.array(bias, dtype=mx.float32),
            mx.array(n, dtype=mx.uint32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        # MLX custom-kernel grids are expressed in threads, rather than blocks.
        grid=(n, 1, 1),
        threadgroup=(min(_THREADGROUP_SIZE, max(n, 1)), 1, 1),
    )
    return out
