#!/usr/bin/env python3
"""Report what the installed MLX build actually is, on this machine, right now.

This replaces a hardcoded chip table. The table only ever described one laptop: an
M5. Every other Apple silicon part reported "unknown", so the agents' first
instruction -- establish the machine -- produced nothing actionable for most users.

Asking Metal works on every chip and cannot go stale, because it reports the machine
in front of it rather than what someone measured once.

Unlike `hooks/scripts/detect-apple-silicon.py`, this DOES import mlx, so it must never
be called from a hook. Importing mlx initialises Metal and allocates.

    python3 probe_mlx_env.py           # human readable
    python3 probe_mlx_env.py --json    # machine readable
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys

# The Neural Accelerator gate, transcribed from is_nax_available() in
# mlx/backend/metal/device.cpp. BOTH halves matter: an M5 on macOS 26.0 has no NAX,
# so a chip-name check alone is wrong.
NAX_MIN_MACOS = (26, 2)


def macos_version() -> tuple[int, ...]:
    raw = platform.mac_ver()[0]
    try:
        return tuple(int(p) for p in raw.split(".")) if raw else ()
    except ValueError:
        return ()


def arch_generation(architecture: str) -> int | None:
    """Extract the generation from a Metal architecture string like applegpu_g17g."""
    m = re.search(r"g(\d+)", architecture or "")
    return int(m.group(1)) if m else None


def nax_status(architecture: str, macos: tuple[int, ...]) -> dict:
    gen = arch_generation(architecture)
    # The suffix selects the threshold: 18 for a 'p' architecture, else 17.
    required = 18 if (architecture or "").endswith("p") else 17
    os_ok = bool(macos) and macos[: len(NAX_MIN_MACOS)] >= NAX_MIN_MACOS
    gen_ok = gen is not None and gen >= required

    if gen is None:
        reason = f"could not parse a generation from architecture {architecture!r}"
    elif not os_ok:
        reason = (
            f"macOS {'.'.join(map(str, macos))} is below "
            f"{'.'.join(map(str, NAX_MIN_MACOS))}; the kernels are compiled in but "
            "will not be dispatched"
        )
    elif not gen_ok:
        reason = f"architecture generation {gen} is below the required {required}"
    else:
        reason = "macOS and architecture generation both satisfy the gate"

    return {
        "available": None if gen is None else (os_ok and gen_ok),
        "reason": reason,
        "architecture": architecture,
        "arch_generation": gen,
        "required_generation": required,
        "macos_ok": os_ok,
    }


def collect() -> dict:
    try:
        import mlx.core as mx
    except ImportError as exc:
        return {"error": f"mlx is not importable: {exc}"}

    info = dict(mx.device_info())
    macos = macos_version()
    out = {
        "mlx_version": mx.__version__,
        "metal_available": bool(mx.metal.is_available()),
        "macos": ".".join(map(str, macos)),
        "python": ".".join(map(str, sys.version_info[:3])),
        "device_info": {k: v for k, v in info.items()},
        "nax": nax_status(str(info.get("architecture", "")), macos),
    }
    try:
        from importlib.metadata import PackageNotFoundError, version

        for pkg in ("mlx", "mlx-lm", "mlx-metal"):
            try:
                out.setdefault("packages", {})[pkg] = version(pkg)
            except PackageNotFoundError:
                out.setdefault("packages", {})[pkg] = None
    except ImportError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = collect()
    if args.json:
        print(json.dumps(d, indent=2, default=str))
        return 0 if "error" not in d else 1

    if "error" in d:
        print(d["error"], file=sys.stderr)
        return 1

    di = d["device_info"]
    mem = di.get("memory_size")
    print(f"MLX {d['mlx_version']}  |  macOS {d['macos']}  |  Python {d['python']}")
    print(f"Device: {di.get('device_name')}  ({di.get('architecture')})")
    if mem:
        print(f"Unified memory: {int(mem) / 1024**3:.0f} GiB")
        rec = di.get("max_recommended_working_set_size")
        if rec:
            print(f"Max recommended working set: {int(rec) / 1024**3:.0f} GiB")

    nax = d["nax"]
    label = {True: "AVAILABLE", False: "unavailable", None: "unknown"}[nax["available"]]
    print(f"\nNeural Accelerators (NAX): {label} -- {nax['reason']}")
    if nax["available"]:
        print(
            "  Reached implicitly through mx.matmul, mx.addmm, mx.quantized_matmul and\n"
            "  mx.fast.scaled_dot_product_attention. There is no way to call them\n"
            "  directly, and no runtime switch to disable them for an A/B."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
