import mlx.core as mx


def rope_batched(x, offsets) -> mx.array:
    """Apply RoPE with an independent starting offset for each batch element."""
    return mx.fast.rope(
        x,
        x.shape[-1],
        traditional=False,
        base=10000.0,
        scale=1.0,
        offset=offsets,
    )
