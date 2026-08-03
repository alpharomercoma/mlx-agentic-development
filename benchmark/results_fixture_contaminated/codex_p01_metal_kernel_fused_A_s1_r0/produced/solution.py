"""Fused scale-and-bias operation implemented with an MLX Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel compiles a Metal library, so it is deliberately
# shared by every call rather than created inside ``fused_scale_bias``.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["inp", "scale", "bias", "n"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < n) {
            out[i] = inp[i] * scale + bias;
        }
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using the fused custom GPU kernel."""
    n = x.size
    if n == 0:
        # There is no work to dispatch, but preserve the required metadata.
        return mx.empty(x.shape, dtype=x.dtype)

    threads = min(256, n)
    (out,) = _SCALE_BIAS_KERNEL(
        inputs=[
            x,
            mx.array(scale, dtype=mx.float32),
            mx.array(bias, dtype=mx.float32),
            mx.array(n, dtype=mx.uint32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        # ``grid`` is expressed in threads (not threadgroups).
        grid=(n, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    return out
