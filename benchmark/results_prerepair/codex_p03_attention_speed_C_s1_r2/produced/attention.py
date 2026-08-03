import math

import mlx.core as mx


def attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Multi-head scaled dot-product attention with a causal mask.

    q: [B, H, T, D]
    k: [B, H, S, D]
    v: [B, H, S, D]
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    t, s = q.shape[2], k.shape[2]
    # Causal mask aligned to the lower right, so the last query attends to the last key.
    row = mx.arange(t).reshape(t, 1) + (s - t)
    col = mx.arange(s).reshape(1, s)
    mask = col > row
    scores = mx.where(mask, mx.array(-float("inf"), dtype=scores.dtype), scores)

    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    return weights @ v
