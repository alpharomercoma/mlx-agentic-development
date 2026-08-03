"""Report MLX GPU memory statistics."""

import mlx.core as mx


def memory_report() -> dict:
    """Return the current MLX memory counters and GPU device name."""
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "device_name": str(mx.metal.device_info()["device_name"]),
    }
