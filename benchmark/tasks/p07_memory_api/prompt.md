Create `solution.py` exposing:

    memory_report() -> dict

It must return a dictionary with exactly these keys:

    "active_bytes"   currently active memory, in bytes (int)
    "peak_bytes"     peak memory since the process started, in bytes (int)
    "cache_bytes"    memory held in MLX's buffer cache, in bytes (int)
    "device_name"    the GPU device name reported by MLX (str)

Use the current, non-deprecated MLX API for each of these. The code must not emit any
deprecation notice when imported or called.
