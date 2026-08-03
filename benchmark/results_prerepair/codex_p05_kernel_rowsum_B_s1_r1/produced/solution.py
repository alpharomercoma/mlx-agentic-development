"""Row-wise float32 reduction implemented with an MLX Metal kernel."""

import mlx.core as mx


# One threadgroup handles one row.  Keeping this a power of two makes the
# threadgroup-memory reduction below straightforward, while each thread still
# processes multiple elements for wide rows.
_THREADS_PER_ROW = 256

_ROW_SUM_KERNEL = mx.fast.metal_kernel(
    name="row_sum_threadgroup_reduction",
    input_names=["x"],
    output_names=["out"],
    source=r"""
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        uint width = x_shape[1];

        // Each lane accumulates a strided portion of this row first.
        float value = 0.0f;
        for (uint col = lane; col < width; col += THREADS_PER_ROW) {
            value += x[row * width + col];
        }

        threadgroup float partial[THREADS_PER_ROW];
        partial[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree-reduce the per-lane partial sums within the row's threadgroup.
        for (uint offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
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
    """Return the float32 sum of each row of a two-dimensional MLX array."""
    if x.ndim != 2:
        raise ValueError("row_sum expects a 2-D array")
    if x.dtype != mx.float32:
        raise TypeError("row_sum expects a float32 array")

    rows = x.shape[0]
    if rows == 0:
        return mx.zeros((0,), dtype=mx.float32)

    outputs = _ROW_SUM_KERNEL(
        inputs=[x],
        template=[("THREADS_PER_ROW", _THREADS_PER_ROW)],
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
        # There is one 256-thread threadgroup for every row.
        grid=(_THREADS_PER_ROW, rows, 1),
        threadgroup=(_THREADS_PER_ROW, 1, 1),
    )
    return outputs[0]
