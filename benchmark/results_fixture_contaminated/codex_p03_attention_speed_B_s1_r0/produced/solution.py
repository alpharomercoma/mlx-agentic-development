import math

import mlx.core as mx


def attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Multi-head scaled dot-product attention with a lower-right causal mask."""
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask="causal"
    )
