"""Fused causal scaled dot-product attention."""

import math

import mlx.core as mx


def attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Multi-head scaled dot-product attention with a lower-right causal mask.

    The fused MLX operation performs the score calculation, float32 softmax, mask,
    and value projection without materialising the ``[B, H, T, S]`` score matrix.
    ``mask="causal"`` has MLX's lower-right causal alignment, matching the
    reference implementation when ``T`` and ``S`` differ.
    """
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask="causal"
    )
