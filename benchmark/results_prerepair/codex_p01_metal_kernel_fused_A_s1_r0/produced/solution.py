"""Fused float32 scale-and-bias operation implemented as a Metal kernel."""

import mlx.core as mx


# Constructing a metal_kernel creates (and potentially JIT-compiles) a Metal
# library, so keep this at module scope and reuse it for every invocation.
_SCALE_BIAS_KERNEL = mx.fast.metal_kernel(
    name="fused_scale_bias_f32",
    input_names=["x", "scale", "bias"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;

        // A one-thread dispatch is used for empty arrays.  Compute the number
        // of elements from the supplied shape so that thread does no write.
        uint n = 1;
        for (int dim = 0; dim < x_ndim; ++dim) {
            n *= x_shape[dim];
        }

        if (elem < n) {
            out[elem] = x[elem] * scale[0] + bias[0];
        }
    """,
)


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU-kernel dispatch."""
    n = x.size
    outputs = _SCALE_BIAS_KERNEL(
        # Scalar buffers carry the per-call values; the elementwise arithmetic
        # itself is entirely in the Metal source above.
        inputs=[
            x,
            mx.array(scale, dtype=mx.float32),
            mx.array(bias, dtype=mx.float32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
        grid=(max(1, n), 1, 1),
        threadgroup=(min(256, max(1, n)), 1, 1),
    )
    return outputs[0]
