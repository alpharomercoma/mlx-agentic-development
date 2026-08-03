import mlx.core as mx


def rope_batched(x, offsets):
    # offset accepts an array of per-sequence positions, so the whole batch is one
    # fused call rather than a Python loop over batch elements.
    return mx.fast.rope(
        x, x.shape[-1], traditional=False, base=10000.0, scale=1.0, offset=offsets
    )
