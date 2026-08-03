"""A row-wise float32 reduction implemented as an MLX Metal kernel."""

import mlx.core as mx


_THREADS_PER_ROW = 256

# One threadgroup is assigned to each row.  Every thread accumulates a strided
# subset of that row, then the threadgroup cooperatively reduces the partials.
_ROW_SUM_KERNEL = mx.fast.metal_kernel(
    name="row_sum_shared",
    input_names=["x"],
    output_names=["out"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.y;
        uint width = x_shape[1];

        float value = 0.0f;
        uint base = row * width;
        for (uint col = lane; col < width; col += 256) {
            value += x[base + col];
        }

        threadgroup float partial[256];
        partial[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) {
                partial[lane] += partial[lane + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            out[row] = partial[0];
        }
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Return the float32 sum of each row of a 2-D float32 MLX array."""
    if x.ndim != 2:
        raise ValueError("row_sum expects a 2-D array")
    if x.dtype != mx.float32:
        raise TypeError("row_sum expects a float32 array")

    rows = x.shape[0]
    # A zero-sized dispatch is invalid; no reduction is needed in this case.
    if rows == 0:
        return mx.zeros((0,), dtype=mx.float32)

    return _ROW_SUM_KERNEL(
        inputs=[x],
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
        grid=(rows, _THREADS_PER_ROW, 1),
        threadgroup=(1, _THREADS_PER_ROW, 1),
    )[0]
