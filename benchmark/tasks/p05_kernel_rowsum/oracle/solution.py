import mlx.core as mx

THREADS_PER_ROW = 256

# atomic_outputs makes the output `device atomic<float>*`. Because outputs are
# otherwise uninitialised, init_value=0 is mandatory for an accumulating kernel.
_kernel = mx.fast.metal_kernel(
    name="row_sum",
    input_names=["x", "n"],
    output_names=["out"],
    source="""
        uint row = threadgroup_position_in_grid.y;
        uint lane = thread_position_in_threadgroup.x;
        uint width = threads_per_threadgroup.x;

        float partial = 0.0f;
        for (uint c = lane; c < n; c += width) {
            partial += x[row * n + c];
        }
        // Reduce within the simdgroup first; atomics alone are far slower.
        partial = simd_sum(partial);
        if (simd_is_first()) {
            atomic_fetch_add_explicit(&out[row], partial, memory_order_relaxed);
        }
    """,
    atomic_outputs=True,
)


def row_sum(x):
    m, n = x.shape
    (out,) = _kernel(
        inputs=[x, mx.array(n, dtype=mx.uint32)],
        output_shapes=[(m,)],
        output_dtypes=[mx.float32],
        # grid is in THREADS: m rows x THREADS_PER_ROW threads each.
        grid=(THREADS_PER_ROW, m, 1),
        threadgroup=(THREADS_PER_ROW, 1, 1),
        init_value=0,
    )
    return out
