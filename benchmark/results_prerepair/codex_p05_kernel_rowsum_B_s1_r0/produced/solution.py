"""Row-wise float32 summation implemented with an MLX Metal kernel."""

import mlx.core as mx


# One threadgroup produces one output element.  Keeping this fixed makes the
# threadgroup scratch array statically sized, while each lane can still process
# arbitrarily many columns with a strided loop.
_THREADS_PER_ROW = 256

_ROW_SUM_KERNEL = mx.fast.metal_kernel(
    name="row_sum_shared_reduction",
    input_names=["x"],
    output_names=["out"],
    source=r"""
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint n = x_shape[1];

        float value = 0.0f;
        uint row_start = row * n;
        for (uint col = lane; col < n; col += 256) {
            value += x[row_start + col];
        }

        threadgroup float partial[256];
        partial[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce the per-lane partial sums in shared threadgroup memory.
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            out[row] = partial[0];
        }
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Return the float32 sum of every row of the two-dimensional array *x*."""
    m = x.shape[0]

    # Dispatching a zero-sized grid is invalid; no reduction is needed when
    # there are no rows.
    if m == 0:
        return mx.zeros((0,), dtype=mx.float32)

    return _ROW_SUM_KERNEL(
        inputs=[x],
        output_shapes=[(m,)],
        output_dtypes=[mx.float32],
        grid=(m * _THREADS_PER_ROW, 1, 1),
        threadgroup=(_THREADS_PER_ROW, 1, 1),
    )[0]
