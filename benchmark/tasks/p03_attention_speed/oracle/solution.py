import math

import mlx.core as mx


def attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    # "causal" uses lower-right alignment, matching the reference mask.
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
