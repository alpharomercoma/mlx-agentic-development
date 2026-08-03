#!/usr/bin/env python3
"""Report which mx.metal.* functions are ACTUALLY deprecated in the installed MLX.

This replaces a hardcoded migration table. The table is right today and will rot; the
deprecation set moves with every release, and functions get removed outright.

The mechanism matters and is the reason a naive check fails: MLX prints its
deprecation notices to **stderr from C++**, not through Python's `warnings` module.
`warnings.catch_warnings()` sees nothing. Only a subprocess with captured stderr can
observe them, which is what this does.

    python3 check_deprecations.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap

# Candidates to test: the ones that moved to top level, plus two that did not, so the
# script demonstrates it can tell the difference rather than flagging everything.
CANDIDATES = [
    ("get_active_memory", "()"),
    ("get_peak_memory", "()"),
    ("reset_peak_memory", "()"),
    ("get_cache_memory", "()"),
    ("clear_cache", "()"),
    ("device_info", "()"),
    ("is_available", "()"),
    ("set_memory_limit", "(2**30)"),
    ("set_cache_limit", "(2**30)"),
    ("set_wired_limit", "(0)"),
]


def probe_one(name: str, call: str) -> dict:
    """Call mx.metal.<name> in a fresh process and capture stderr."""
    script = textwrap.dedent(f"""
        import mlx.core as mx
        fn = getattr(mx.metal, {name!r}, None)
        if fn is None:
            print("ABSENT")
        else:
            try:
                fn{call}
                print("CALLED")
            except Exception as exc:
                print("RAISED:" + str(exc)[:80])
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr
    deprecated = "deprecated" in stderr.lower()
    replacement = None
    if deprecated:
        for line in stderr.splitlines():
            if "deprecated" in line.lower() and " Use " in line:
                replacement = line.split(" Use ", 1)[1].strip().rstrip(".")
                break
    return {
        "name": name,
        "present": stdout != "ABSENT",
        "deprecated": deprecated,
        "replacement": replacement,
        "status": stdout[:60],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import mlx.core as mx
    except ImportError as exc:
        print(f"mlx is not importable: {exc}", file=sys.stderr)
        return 1

    rows = [probe_one(n, c) for n, c in CANDIDATES]
    if args.json:
        print(json.dumps({"mlx_version": mx.__version__, "functions": rows}, indent=2))
        return 0

    print(f"MLX {mx.__version__}  --  mx.metal.* deprecation status\n")
    print(f"{'function':22s} {'present':8s} {'deprecated':11s} use instead")
    print("-" * 78)
    for r in rows:
        print(
            f"mx.metal.{r['name']:13s} {str(r['present']):8s} "
            f"{str(r['deprecated']):11s} {r['replacement'] or '-'}"
        )
    dep = [r["name"] for r in rows if r["deprecated"]]
    ok = [r["name"] for r in rows if r["present"] and not r["deprecated"]]
    print(
        f"\nDeprecated ({len(dep)}): {', '.join(dep) or 'none'}"
        f"\nStill current ({len(ok)}): {', '.join(ok) or 'none'}"
        "\n\nThese notices go to stderr from C++, not through Python warnings, so"
        "\nwarnings.catch_warnings() cannot see them and a test that only inspects"
        "\nwarnings will pass against deprecated code."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
