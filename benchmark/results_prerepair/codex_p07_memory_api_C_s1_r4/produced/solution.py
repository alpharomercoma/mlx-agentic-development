"""Current MLX GPU memory reporting helpers."""

import mlx.core as mx


def memory_report() -> dict:
    """Return MLX's current GPU memory counters and device name."""
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "device_name": str(mx.device_info()["device_name"]),
    }
