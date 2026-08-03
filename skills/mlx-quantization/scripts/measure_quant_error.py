#!/usr/bin/env python3
"""Measure the quantisation error separation on THIS machine, at THIS version.

This replaces two frozen numbers (7.7e-4 for 4-bit affine, 4.1e-4 for mxfp4). Frozen
tolerances are the worst kind of documentation: they look authoritative, they are
invisible when wrong, and upstream moves them. A commit fixing nvfp4 quantized_matmul
through the split-K path landed within a month of the version those numbers came from.

What it prints, per mode:

  qmm_vs_dequant    a CORRECT quantised matmul against dequantise-then-matmul
  full_vs_dequant   an UNQUANTISED matmul against the same reference

The gap between them is what a tolerance has to sit inside. Choose a threshold from
the measurement -- comfortably above the correct implementation, comfortably below the
unquantised one -- rather than picking a round number and hoping.

    python3 measure_quant_error.py
"""

from __future__ import annotations

import argparse
import json
import sys

SHAPES = ((128, 256), (64, 128), (256, 512))


def rel_frobenius(got, exp) -> float:
    import mlx.core as mx

    return (mx.sqrt(mx.sum((got - exp) ** 2)) / mx.sqrt(mx.sum(exp**2))).item()


def measure(mode: str, group_size: int | None, bits: int | None) -> list[dict]:
    import mlx.core as mx

    kw = {"mode": mode}
    if group_size is not None:
        kw["group_size"] = group_size
    if bits is not None:
        kw["bits"] = bits

    rows = []
    for out_f, in_f in SHAPES:
        mx.random.seed(0)
        w = mx.random.normal((out_f, in_f)).astype(mx.float32)
        x = mx.random.normal((8, in_f)).astype(mx.float32)
        mx.eval(w, x)

        packed = list(mx.quantize(w, **kw))
        deq = mx.dequantize(*packed, **kw, dtype=mx.float32)
        reference = x @ deq.T

        mm_kw = dict(kw)
        qmm = mx.quantized_matmul(
            x,
            packed[0],
            scales=packed[1],
            biases=packed[2] if len(packed) > 2 else None,
            transpose=True,
            **mm_kw,
        )
        full = x @ w.T
        mx.eval(reference, qmm, full)

        rows.append(
            {
                "shape": [out_f, in_f],
                "qmm_vs_dequant": rel_frobenius(qmm, reference),
                "full_vs_dequant": rel_frobenius(full, reference),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError as exc:
        print(f"mlx is not importable: {exc}", file=sys.stderr)
        return 1

    configs = [("affine", 64, 4), ("mxfp4", None, None)]
    out = {}
    for mode, gs, bits in configs:
        try:
            out[mode] = measure(mode, gs, bits)
        except Exception as exc:  # noqa: BLE001
            out[mode] = [{"error": str(exc)[:160]}]

    if args.json:
        print(json.dumps({"mlx_version": mx.__version__, "results": out}, indent=2))
        return 0

    print(f"MLX {mx.__version__}  --  relative Frobenius error\n")
    for mode, rows in out.items():
        if "error" in rows[0]:
            print(f"{mode}: {rows[0]['error']}")
            continue
        corr = max(r["qmm_vs_dequant"] for r in rows)
        wrong = min(r["full_vs_dequant"] for r in rows)
        print(f"{mode}:")
        for r in rows:
            print(
                f"  {str(tuple(r['shape'])):12s} correct {r['qmm_vs_dequant']:.2e}   "
                f"unquantised {r['full_vs_dequant']:.2e}"
            )
        if corr > 0:
            print(
                f"  -> worst correct {corr:.2e}, best unquantised {wrong:.2e}, "
                f"separation {wrong / corr:.0f}x"
            )
            print(
                f"  -> a tolerance anywhere in ({corr:.1e}, {wrong:.1e}) separates them\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
