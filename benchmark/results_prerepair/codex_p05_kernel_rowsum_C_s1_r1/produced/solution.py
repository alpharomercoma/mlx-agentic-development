"""Row-wise float32 reduction implemented with an MLX Metal kernel."""

import mlx.core as mx


# This must stay a compile-time constant because it sizes the threadgroup memory.
_THREADS_PER_ROW = 256


_ROW_SUM_KERNEL = mx.fast.metal_kernel(
    name="row_sum",
    input_names=["x", "n_cols"],
    output_names=["out"],
    source=r"""
        const uint tid = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.x;

        // Each thread accumulates a strided part of this row.  The bound makes
        // arbitrary (including non-power-of-two) row widths safe.
        float partial = 0.0f;
        const uint row_start = row * n_cols;
        for (uint col = tid; col < n_cols; col += 256) {
            partial += x[row_start + col];
        }

        threadgroup float scratch[256];
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A tree reduction combines the 256 partial sums for the row.
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (tid < offset) {
                scratch[tid] += scratch[tid + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            out[row] = scratch[0];
        }
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Return the float32 sum of every row of the two-dimensional array ``x``."""
    if x.ndim != 2:
        raise ValueError("row_sum expects a 2-D array")
    if x.dtype != mx.float32:
        raise TypeError("row_sum expects a float32 array")

    n_rows, n_cols = x.shape
    if n_rows == 0:
        return mx.zeros((0,), dtype=mx.float32)

    (out,) = _ROW_SUM_KERNEL(
        # A 0-D uint32 array is passed to Metal by value, so ``n_cols`` is a
        # uint in the kernel rather than a device pointer.
        inputs=[x, mx.array(n_cols, dtype=mx.uint32)],
        output_shapes=[(n_rows,)],
        output_dtypes=[mx.float32],
        grid=(n_rows * _THREADS_PER_ROW, 1, 1),
        threadgroup=(_THREADS_PER_ROW, 1, 1),
    )
    return out
