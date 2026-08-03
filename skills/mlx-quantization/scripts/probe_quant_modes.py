#!/usr/bin/env python3
"""Print what mx.quantize actually returns, per mode, in the installed MLX.

This replaces a hardcoded table. The trap it documents is real -- `affine` returns
three values and `mxfp4`/`mxfp8`/`nvfp4` return two, and each mode has its own default
group size and bit width -- but a table freezes an answer that upstream changes. MLX
ships roughly every 2.5 weeks, and commits touching nvfp4 scales landed within a month
of the version this kit was written against.

Asking the installed library is always right.

    python3 probe_quant_modes.py
"""

from __future__ import annotations

import argparse
import json
import sys

MODES = ("affine", "mxfp4", "mxfp8", "nvfp4")


def probe(mode: str) -> dict:
    import mlx.core as mx

    w = mx.random.normal((64, 256)).astype(mx.float32)
    mx.eval(w)
    try:
        packed = mx.quantize(w, mode=mode)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole probe
        return {"mode": mode, "supported": False, "error": str(exc)[:160]}

    parts = list(packed)
    entry = {
        "mode": mode,
        "supported": True,
        "returns": len(parts),
        "shapes": [tuple(p.shape) for p in parts],
        "names": ["w_q", "scales", "biases"][: len(parts)],
    }

    # Round-trip to recover the effective group size and confirm dequantize's default
    # output dtype, which infers rather than preserving float32.
    try:
        deq = mx.dequantize(*parts, mode=mode)
        mx.eval(deq)
        entry["dequantize_default_dtype"] = str(deq.dtype)
        entry["effective_group_size"] = w.shape[-1] // parts[1].shape[-1]
    except Exception as exc:  # noqa: BLE001
        entry["dequantize_error"] = str(exc)[:160]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError as exc:
        print(f"mlx is not importable: {exc}", file=sys.stderr)
        return 1

    rows = [probe(m) for m in MODES]
    if args.json:
        print(json.dumps({"mlx_version": mx.__version__, "modes": rows}, indent=2))
        return 0

    print(f"MLX {mx.__version__}\n")
    print(f"{'mode':8s} {'returns':>7s}  {'group':>5s}  {'dequantize dtype':17s} shapes")
    print("-" * 76)
    for r in rows:
        if not r["supported"]:
            print(f"{r['mode']:8s} {'-':>7s}  unsupported: {r['error'][:40]}")
            continue
        print(
            f"{r['mode']:8s} {r['returns']:7d}  "
            f"{r.get('effective_group_size', '?'):>5}  "
            f"{r.get('dequantize_default_dtype', '?'):17s} {r['shapes']}"
        )
    print(
        "\n'returns' is the number of values to unpack. Code written for affine that\n"
        "unpacks three values raises on every other mode, and biases must be omitted\n"
        "from quantized_matmul when the mode does not produce them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
