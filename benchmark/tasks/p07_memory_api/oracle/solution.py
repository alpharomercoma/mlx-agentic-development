import mlx.core as mx


def memory_report():
    # These all moved from mx.metal.* to top-level mx.* and the old spellings now
    # print a deprecation notice to stderr.
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "device_name": str(mx.device_info()["device_name"]),
    }
