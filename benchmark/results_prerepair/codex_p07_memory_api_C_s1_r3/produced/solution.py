"""Current MLX GPU memory statistics."""

import mlx.core as mx


def memory_report() -> dict:
    """Return MLX's current GPU memory counters and device name."""
    device_info = mx.device_info()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "device_name": str(device_info["device_name"]),
    }
