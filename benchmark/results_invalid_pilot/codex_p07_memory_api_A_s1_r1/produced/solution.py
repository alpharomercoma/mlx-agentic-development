"""Current MLX memory and GPU device reporting helpers."""

import mlx.core as mx


def memory_report() -> dict:
    """Return MLX's current process memory counters and GPU name."""
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "device_name": str(mx.device_info()["device_name"]),
    }
