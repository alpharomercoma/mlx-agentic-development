"""Row-wise float32 summation implemented with an MLX Metal kernel."""

import mlx.core as mx


# One complete SIMD group is assigned to each row.  This makes the reduction
# independent of the row width while ensuring every row is processed by many
# GPU threads.
_THREADS_PER_ROW = 32

_ROW_SUM_KERNEL = mx.fast.metal_kernel(
    name="row_sum_simd",
    input_names=["x"],
    output_names=["out"],
    source=r"""
        // The dispatch contains exactly one 32-thread threadgroup per row.
        uint lane = thread_index_in_simdgroup;
        uint row = thread_position_in_grid.x / 32;

        float partial = 0.0f;
        uint base = row * N;
        for (uint col = lane; col < N; col += 32) {
            partial += x[base + col];
        }

        // All lanes in this SIMD group participate in the row reduction.
        partial = simd_sum(partial);
        if (lane == 0) {
            out[row] = partial;
        }
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Return the float32 sum of every row of the 2-D array *x*.

    The input is made row-contiguous by MLX's custom-kernel wrapper when
    necessary, so the kernel can index each row directly.
    """
    if x.ndim != 2:
        raise ValueError("row_sum expects a 2-D array")
    if x.dtype != mx.float32:
        raise TypeError("row_sum expects a float32 array")

    rows, columns = x.shape
    if rows == 0:
        return mx.zeros((0,), dtype=mx.float32)

    return _ROW_SUM_KERNEL(
        inputs=[x],
        template=[("N", columns)],
        grid=(rows * _THREADS_PER_ROW, 1, 1),
        threadgroup=(_THREADS_PER_ROW, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )[0]
