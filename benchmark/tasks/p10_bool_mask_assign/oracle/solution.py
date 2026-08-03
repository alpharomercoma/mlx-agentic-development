import mlx.core as mx


def safe_take(x, idx, fill):
    n = x.shape[0]
    # Wrap negatives first, then decide validity. MLX performs no bounds checking:
    # an out-of-range gather is undefined behaviour that reads unowned memory
    # rather than raising, so indices must be clamped BEFORE the gather and the
    # invalid positions replaced afterwards.
    wrapped = mx.where(idx < 0, idx + n, idx)
    valid = (wrapped >= 0) & (wrapped < n)
    safe = mx.where(valid, wrapped, mx.zeros_like(wrapped))
    gathered = mx.take(x, safe)
    return mx.where(valid, gathered, mx.array(fill, dtype=x.dtype)).astype(mx.float32)
