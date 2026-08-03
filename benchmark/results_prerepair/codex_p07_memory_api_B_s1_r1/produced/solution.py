"""A small, warning-free snapshot of MLX GPU memory usage."""

import mlx.core as mx


def memory_report() -> dict:
    """Return the current MLX memory counters and Metal device name."""
    return {
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "device_name": mx.metal.device_info()["name"],
    }
