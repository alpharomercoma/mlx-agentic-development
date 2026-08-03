"""Fused float32 scale-and-bias operation implemented as a Metal kernel."""

import mlx.core as mx


# Constructing this object JIT-compiles the Metal source, so it deliberately lives
# at module scope rather than in ``fused_scale_bias``.
_FUSED_SCALE_BIAS = mx.fast.metal_kernel(
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


def fused_scale_bias(x: mx.array, scale: float, bias: float) -> mx.array:
    """Return ``x * scale + bias`` using one custom GPU kernel launch.

    ``x`` is expected to be float32.  ``scale`` and ``bias`` are passed as
    by-value scalar kernel inputs, avoiding scalar-array allocation on each call.
    """
    (out,) = _FUSED_SCALE_BIAS(
        inputs=[x, float(scale), float(bias)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        # Keep the dispatch valid for zero-sized shapes; the bounds check means
        # that lone thread performs no access in that case.
        grid=(max(x.size, 1), 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out
