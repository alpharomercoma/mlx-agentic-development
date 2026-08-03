"""Row-wise float32 sum implemented with an MLX custom Metal kernel."""

import mlx.core as mx


# One threadgroup handles one row.  The fixed, power-of-two group size makes
# the in-threadgroup reduction straightforward while allowing every thread to
# accumulate a strided portion of arbitrarily sized rows.
_THREADS_PER_ROW = 256

_row_sum_kernel = mx.fast.metal_kernel(
    name="row_sum_parallel",
    input_names=["x"],
    output_names=["out"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;

        T accum = T(0);
        for (uint col = lane; col < uint(N); col += threads_per_threadgroup.x) {
            accum += x[row * uint(N) + col];
        }

        threadgroup T partial[256];
        partial[lane] = accum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
    """Return the float32 sum of each row of a two-dimensional ``x``.

    ``N`` is specialized into the generated Metal kernel, so the loop's tail
    check handles widths that are not multiples of the threadgroup size.
    """
    m, n = x.shape

    # Launching a zero-sized Metal grid is not meaningful.  There is no input
    # work to reduce in this case, so construct the correctly typed empty
    # result directly.
    if m == 0:
        return mx.zeros((0,), dtype=mx.float32)

    return _row_sum_kernel(
        inputs=[x],
        template=[("T", mx.float32), ("N", n)],
        grid=(m, 1, 1),
        threadgroup=(_THREADS_PER_ROW, 1, 1),
        output_shapes=[(m,)],
        output_dtypes=[mx.float32],
    )[0]
