import mlx.core as mx

# Number of threads cooperating on each row. Must be a power of two for the
# threadgroup tree reduction below.
_TPB = 256

# Build the custom Metal kernel once at module scope (mlx caches the compiled
# library; constructing it per-call would recompile).
_row_sum_kernel = mx.fast.metal_kernel(
    name="row_sum_kernel",
    input_names=["x", "n_cols"],
    output_names=["out"],
    source=f"""
        constexpr uint TPB = {_TPB};
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;

        // Every thread of the row's threadgroup sums a strided subset of the
        // row's columns, so the work within a row is shared across TPB threads.
        // This handles any N (no alignment requirement): each column
        // c in [0, n_cols) is visited by exactly one thread.
        float partial = 0.0f;
        for (uint c = tid; c < n_cols; c += TPB) {{
            partial += x[row * n_cols + c];
        }}

        // Tree reduction of the TPB partial sums within the threadgroup.
        threadgroup float partials[TPB];
        partials[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TPB / 2; s > 0; s >>= 1) {{
            if (tid < s) {{
                partials[tid] += partials[tid + s];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            out[row] = partials[0];
        }}
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Sum each row of a 2-D float32 array with a custom GPU kernel.

    Args:
        x: 2-D float32 array of shape [M, N].

    Returns:
        1-D float32 array of shape [M] where element i is the sum of row i.
    """
    M, N = x.shape
    if M == 0:
        return mx.zeros((0,), mx.float32)

    (out,) = _row_sum_kernel(
        inputs=[x, mx.array(N, dtype=mx.uint32)],
        output_shapes=[(M,)],
        output_dtypes=[mx.float32],
        # One threadgroup per row; _TPB threads cooperate on each row.
        grid=(_TPB, M, 1),
        threadgroup=(_TPB, 1, 1),
    )
    return out
