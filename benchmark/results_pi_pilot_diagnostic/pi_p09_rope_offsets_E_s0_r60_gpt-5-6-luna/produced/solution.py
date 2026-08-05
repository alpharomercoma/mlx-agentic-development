import mlx.core as mx


def rope_batched(x, offsets):
    """Apply rotary positional embedding with one starting offset per batch item."""
    return mx.fast.rope(
        x,
        x.shape[-1],
        traditional=False,
        base=10000.0,
        scale=1.0,
        offset=offsets,
    )
