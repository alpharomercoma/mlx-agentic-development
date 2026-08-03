#!/usr/bin/env python3
"""Canonical Apple silicon constants for mlx-agentic-development.

This file is the single source of truth for every chip figure in this repository.
Reference pages are generated from it (see .github/scripts/gen_hardware_doc.py) and
CI fails if they drift. Never hardcode a chip spec anywhere else.

It must never import mlx. Importing mlx initializes Metal and allocates GPU
resources, which would disturb any workload the user already has running -- and this
module is executed by a SessionStart hook on every session.

Run directly for JSON:  python3 detect-apple-silicon.py
Human-readable:         python3 detect-apple-silicon.py --text
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Chip table
#
# `metal_arch` is the string MLX reports as device_info()["architecture"]. It is
# the input to the NAX gate, and the *only* reliable way to obtain it is to ask
# Metal -- there is no sysctl for it. So it is recorded here per chip, and each
# row carries how it was established.
#
# verified="executed"  -> observed on real hardware by this project, date noted
# verified="apple"     -> stated in Apple published specifications
# verified="unknown"   -> not established; consumers must treat as unknown
# ---------------------------------------------------------------------------

CHIPS: dict[str, dict] = {
    "M5": {
        "metal_arch": "applegpu_g17g",  # executed 2026-08-04, MacBook Air M5
        "arch_gen": 17,
        "gpu_cores": [8, 10],
        "memory_bandwidth_gbs": 153,  # apple newsroom, M5 announcement
        "unified_memory_gb": [16, 24, 32],
        "neural_engine_cores": 16,
        "has_neural_accelerators": True,  # "a Neural Accelerator in each core"
        "verified": "executed",
    },
    # Older chips are deliberately left unpopulated rather than guessed. Their
    # Metal architecture strings were not observed by this project, and the NAX
    # gate reads that string. A wrong arch_gen here would silently mis-report
    # accelerator availability, which is worse than reporting "unknown".
    "M4": {"arch_gen": None, "has_neural_accelerators": False, "verified": "unknown"},
    "M3": {"arch_gen": None, "has_neural_accelerators": False, "verified": "unknown"},
    "M2": {"arch_gen": None, "has_neural_accelerators": False, "verified": "unknown"},
    "M1": {"arch_gen": None, "has_neural_accelerators": False, "verified": "unknown"},
}

# Minimum macOS for MLX's Neural Accelerator (NAX) kernels, from the runtime gate
# in mlx/backend/metal/device.cpp::is_nax_available().
NAX_MIN_MACOS = (26, 2)

# MLX's own requirements, from docs/src/install.rst.
MLX_MIN_MACOS = (14, 0)
MLX_MIN_PYTHON = (3, 10)


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def detect_chip() -> str | None:
    """Return the chip family name, e.g. "M5", or None if not Apple silicon."""
    brand = _sysctl("machdep.cpu.brand_string") or ""
    # "Apple M5", "Apple M4 Pro", "Apple M2 Max", "Apple M1 Ultra"
    m = re.match(r"Apple (M\d+)", brand)
    return m.group(1) if m else None


def detect_variant(chip: str | None) -> str | None:
    """Return "Pro", "Max", "Ultra", or None for the base part."""
    if not chip:
        return None
    brand = _sysctl("machdep.cpu.brand_string") or ""
    m = re.search(rf"Apple {chip}\s+(Pro|Max|Ultra)\b", brand)
    return m.group(1) if m else None


def macos_version() -> tuple[int, ...] | None:
    raw = platform.mac_ver()[0]
    if not raw:
        return None
    try:
        return tuple(int(p) for p in raw.split("."))
    except ValueError:
        return None


def _at_least(actual: tuple[int, ...] | None, minimum: tuple[int, ...]) -> bool:
    if actual is None:
        return False
    # Compare only as many components as `minimum` specifies, so 26.5.2 >= (26, 2).
    return actual[: len(minimum)] >= minimum


def nax_status(chip: str | None, macos: tuple[int, ...] | None) -> dict:
    """Evaluate MLX's Neural Accelerator gate.

    Faithful to is_nax_available() in mlx/backend/metal/device.cpp:

        can_use_nax  = macOS >= 26.2
        arch         = last character of the Metal architecture string
        gen          = architecture generation
        can_use_nax &= gen >= (arch == 'p' ? 18 : 17)

    Both halves matter. A model that checks only the chip name will claim NAX on
    an M5 running macOS 26.0, where it is silently unavailable.
    """
    info = CHIPS.get(chip or "", {})
    arch = info.get("metal_arch")
    gen = info.get("arch_gen")

    os_ok = _at_least(macos, NAX_MIN_MACOS)

    if gen is None or arch is None:
        return {
            "available": None,
            "reason": "metal architecture for this chip is not recorded in the chip "
            "table; query mx.device_info()['architecture'] to establish it",
            "macos_ok": os_ok,
            "required_macos": ".".join(str(p) for p in NAX_MIN_MACOS),
        }

    required_gen = 18 if arch.endswith("p") else 17
    gen_ok = gen >= required_gen

    if os_ok and gen_ok:
        reason = "macOS and GPU architecture generation both satisfy the gate"
    elif not os_ok:
        reason = (
            f"macOS is below {'.'.join(str(p) for p in NAX_MIN_MACOS)}; NAX kernels "
            "are compiled in but will not be dispatched"
        )
    else:
        reason = f"GPU architecture generation {gen} is below the required {required_gen}"

    return {
        "available": bool(os_ok and gen_ok),
        "reason": reason,
        "macos_ok": os_ok,
        "arch_gen_ok": gen_ok,
        "metal_arch": arch,
        "arch_gen": gen,
        "required_arch_gen": required_gen,
        "required_macos": ".".join(str(p) for p in NAX_MIN_MACOS),
    }


def collect() -> dict:
    chip = detect_chip()
    variant = detect_variant(chip)
    macos = macos_version()
    info = CHIPS.get(chip or "", {})

    mem_bytes = _sysctl("hw.memsize")
    try:
        mem_gb = round(int(mem_bytes) / (1024**3)) if mem_bytes else None
    except ValueError:
        mem_gb = None

    return {
        "is_apple_silicon": chip is not None,
        "chip": chip,
        "variant": variant,
        # Join on the parts that exist -- an f-string here renders "M5 None"
        # for a base part, since variant is None rather than "".
        "chip_label": " ".join(p for p in (chip, variant) if p) or None,
        "macos": ".".join(str(p) for p in macos) if macos else None,
        "macos_supports_mlx": _at_least(macos, MLX_MIN_MACOS),
        "unified_memory_gb": mem_gb,
        "memory_bandwidth_gbs": info.get("memory_bandwidth_gbs"),
        "neural_engine_cores": info.get("neural_engine_cores"),
        "constants_verified": info.get("verified", "unknown"),
        "nax": nax_status(chip, macos),
        "python": ".".join(str(p) for p in sys.version_info[:3]),
        "python_supports_mlx": sys.version_info[:2] >= MLX_MIN_PYTHON,
    }


def as_text(d: dict) -> str:
    if not d["is_apple_silicon"]:
        return "Not Apple silicon. MLX requires an Apple silicon Mac."

    lines = [
        f"{d['chip_label']}, macOS {d['macos']}, {d['unified_memory_gb']} GB unified"
    ]
    if d["memory_bandwidth_gbs"]:
        lines[0] += f", {d['memory_bandwidth_gbs']} GB/s"

    nax = d["nax"]
    if nax["available"] is True:
        lines.append(
            f"Neural Accelerators (NAX): AVAILABLE ({nax['metal_arch']}, gen "
            f"{nax['arch_gen']}). Reached implicitly through mx.matmul, mx.addmm, "
            "mx.quantized_matmul, and mx.fast.scaled_dot_product_attention -- never "
            "called directly."
        )
    elif nax["available"] is False:
        lines.append(f"Neural Accelerators (NAX): unavailable -- {nax['reason']}")
    else:
        lines.append(f"Neural Accelerators (NAX): unknown -- {nax['reason']}")

    if d["constants_verified"] != "executed":
        lines.append(
            f"Chip constants for this part are marked '{d['constants_verified']}'; "
            "treat figures as unverified."
        )
    if not d["python_supports_mlx"]:
        lines.append(
            f"Python {d['python']} is below MLX's minimum "
            f"{'.'.join(str(p) for p in MLX_MIN_PYTHON)}."
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", action="store_true", help="human-readable output")
    args = ap.parse_args()

    data = collect()
    print(as_text(data) if args.text else json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
