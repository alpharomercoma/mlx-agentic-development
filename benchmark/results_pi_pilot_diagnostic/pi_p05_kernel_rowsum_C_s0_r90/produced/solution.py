import mlx.core as mx

# Threads per row. A power of two so the threadgroup tree reduction
# can halve its stride repeatedly. Each row is handled by a whole
# threadgroup of _T threads, so the per-row work is shared.
_THREADS = 256

# Build the Metal kernel once at module scope. Every construction
# JIT-compiles a new Metal library, so building inside the hot
# function would dominate runtime.
_kernel = mx.fast.metal_kernel(
    name="row_sum",
    input_names=["x", "ncols"],
    output_names=["out"],
    source="""
        const uint T = 256;               // threads per row (== _THREADS)
        const uint gid = thread_position_in_grid.x;
        const uint row = gid / T;         // which row this threadgroup owns
        const uint lane = gid % T;        // position inside the threadgroup
        const uint n = (uint)ncols[0];    // row length (1-d array binds as pointer)

        // Strided partial sum: each thread accumulates every T-th element,
        // so a row of any length (not a multiple of T) is fully covered.
        float s = 0.0f;
        const uint base = row * n;
        for (uint i = lane; i < n; i += T) {
            s += x[base + i];
        }

        // Threadgroup tree reduction of the T partial sums.
        threadgroup float tg[256];
        tg[lane] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                tg[lane] += tg[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            out[row] = tg[0];
        }
    """,
)


def row_sum(x: mx.array) -> mx.array:
    """Return a 1-D float32 array of shape [M] with the sum of each row of x.

    x must be a 2-D float32 array of shape [M, N]. The reduction runs on
    the GPU via a custom Metal kernel: each row is reduced by a threadgroup
    of _THREADS threads (strided accumulation + threadgroup tree reduction).
    """
    if x.ndim != 2:
        raise ValueError(f"row_sum expects a 2-D array, got shape {x.shape}")
    if x.dtype != mx.float32:
        raise ValueError(f"row_sum expects float32, got {x.dtype}")

    m, n = x.shape

    if m == 0:
        return mx.zeros((0,), mx.float32)

    (out,) = _kernel(
        inputs=[x, mx.array([n], mx.int32)],
        output_shapes=[(m,)],
        output_dtypes=[mx.float32],
        grid=(m * _THREADS, 1, 1),  # THREADS, not threadgroups
        threadgroup=(_THREADS, 1, 1),
    )
    return out
