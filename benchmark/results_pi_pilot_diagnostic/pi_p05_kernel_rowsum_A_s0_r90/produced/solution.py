import mlx.core as mx

# Reduction performed by a custom Metal kernel, one threadgroup per row.
# The THREADS threads assigned to a row share the work: each thread strides
# over a subset of the row's columns accumulating a private partial sum, the
# partials are combined with a tree reduction in threadgroup memory, and the
# row's first thread writes the final total. No built-in reduction operators
# are used, and N may be arbitrary (threads whose column index exceeds N just
# contribute 0).
_THREADS = 512

_row_sum_kernel = mx.fast.metal_kernel(
    name="row_sum_kernel",
    input_names=["inp"],
    output_names=["out"],
    source=f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint nthreads = threads_per_threadgroup.x;

        // Strided pass over the row: each thread sums its own columns.
        float acc = 0.0;
        for (uint j = tid; j < N; j += nthreads) {{
            acc += inp[row * N + j];
        }}

        // Tree reduction of the per-thread partials in shared memory.
        threadgroup float smem[{_THREADS}];
        smem[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = nthreads / 2; s > 0; s >>= 1) {{
            if (tid < s) {{
                smem[tid] += smem[tid + s];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            out[row] = smem[0];
        }}
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Sum each row of the 2-D float32 array ``x`` using a custom GPU kernel.

    Args:
        x: 2-D float32 array of shape [M, N].

    Returns:
        float32 array of shape [M] where element i is the sum of row i.
    """
    M, N = x.shape
    return _row_sum_kernel(
        inputs=[x],
        template=[("N", N)],
        grid=(M * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(M,)],
        output_dtypes=[mx.float32],
    )[0]
