import mlx.core as mx

# Built once at module scope: each construction creates and JIT-compiles a new
# Metal library, so building inside the function would dominate runtime.
_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i < n) {
        out[i] = inp[i] * scale + bias;
    }
"""

_kernel = mx.fast.metal_kernel(
    name="fused_scale_bias",
    input_names=["inp", "scale", "bias", "n"],
    output_names=["out"],
    source=_SOURCE,
)


def fused_scale_bias(x, scale, bias):
    n = x.size
    tg = 256
    (out,) = _kernel(
        inputs=[
            x,
            mx.array(scale, dtype=mx.float32),
            mx.array(bias, dtype=mx.float32),
            mx.array(n, dtype=mx.uint32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        # grid is in THREADS, not threadblocks.
        grid=(n, 1, 1),
        threadgroup=(min(tg, max(n, 1)), 1, 1),
    )
    return out
