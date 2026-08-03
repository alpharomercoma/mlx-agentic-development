#!/usr/bin/env python3
"""Show how mx.fast.metal_kernel binds each input rank, by compiling all six cases.

This replaces a hardcoded table. The rule it documents cost a bare agent an entire
benchmark task: it passed scalars as 0-d arrays and then subscripted them, which are
individually reasonable choices and jointly a compile error.

Rather than asserting the rule, this compiles every combination and reports what the
installed MLX actually does. If upstream changes the binding, the script tells you;
a table would quietly lie.

    python3 probe_kernel_binding.py
"""

from __future__ import annotations

import argparse
import json
import sys

BODY_SUBSCRIPT = "uint i = thread_position_in_grid.x; out[i] = x[i] * scale[0];"
BODY_VALUE = "uint i = thread_position_in_grid.x; out[i] = x[i] * scale;"


def first_error(exc: Exception) -> str:
    lines = [ln.strip() for ln in str(exc).splitlines() if "error:" in ln]
    return (lines[0] if lines else str(exc).splitlines()[0])[:110]


def try_case(name: str, body: str, scale) -> dict:
    import mlx.core as mx

    try:
        kernel = mx.fast.metal_kernel(
            # The kernel name becomes a Metal identifier, so it must be a valid one.
            name=f"probe_{name}",
            input_names=["x", "scale"],
            output_names=["out"],
            source=body,
        )
        (out,) = kernel(
            inputs=[mx.array([1.0, 2.0], dtype=mx.float32), scale],
            output_shapes=[(2,)],
            output_dtypes=[mx.float32],
            grid=(2, 1, 1),
            threadgroup=(2, 1, 1),
        )
        mx.eval(out)
        return {"case": name, "ok": True, "result": out.tolist()}
    except Exception as exc:  # noqa: BLE001 - the failures are the point
        return {"case": name, "ok": False, "error": first_error(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError as exc:
        print(f"mlx is not importable: {exc}", file=sys.stderr)
        return 1

    cases = [
        ("pyfloat_value", BODY_VALUE, 3.0),
        ("pyfloat_subscript", BODY_SUBSCRIPT, 3.0),
        ("zerod_value", BODY_VALUE, mx.array(3.0, dtype=mx.float32)),
        ("zerod_subscript", BODY_SUBSCRIPT, mx.array(3.0, dtype=mx.float32)),
        ("oned_value", BODY_VALUE, mx.array([3.0], dtype=mx.float32)),
        ("oned_subscript", BODY_SUBSCRIPT, mx.array([3.0], dtype=mx.float32)),
    ]
    rows = [try_case(n, b, s) for n, b, s in cases]

    if args.json:
        print(json.dumps({"mlx_version": mx.__version__, "cases": rows}, indent=2))
        return 0

    print(f"MLX {mx.__version__}  --  how each input rank binds in a Metal kernel\n")
    print(f"{'input passed':22s} {'used as':12s} {'result':10s} detail")
    print("-" * 92)
    for r, (n, _, _) in zip(rows, cases, strict=True):
        passed, usage = n.rsplit("_", 1)
        status = "COMPILES" if r["ok"] else "FAILS"
        detail = str(r.get("result", "")) if r["ok"] else r["error"]
        print(f"{passed:22s} {usage:12s} {status:10s} {detail}")

    working = [r["case"] for r in rows if r["ok"]]
    print(
        "\nThe combinations that compile are the binding rule, and it is by RANK:\n"
        f"  {', '.join(working) or 'none'}\n"
        "A Python scalar or a 0-d array binds BY VALUE (use `scale`); a 1-d array\n"
        "binds as a POINTER (use `scale[0]`), even with a single element."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
