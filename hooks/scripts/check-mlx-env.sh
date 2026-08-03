#!/usr/bin/env bash
# SessionStart hook for mlx-agentic-development.
#
# Seeds the detected chip, its NAX status, and the installed MLX versions into the
# model's context, so kernel and performance work starts from the real machine
# rather than from priors.
#
# Two rules govern this script:
#
#   1. It never imports mlx. Importing mlx initializes Metal and allocates GPU
#      resources, which would disturb a workload the user already has running.
#      Package presence and versions are read from distribution metadata instead.
#   2. It emits hookSpecificOutput.additionalContext JSON, never ANSI colour.
#      SessionStart stdout is injected into the model's context; escape codes
#      would land there as literal garbage.
#
# Everything is a warning. A machine without MLX installed gets guidance, not a
# blocked session. Always exits 0.

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DETECT="${SCRIPT_DIR}/detect-apple-silicon.py"

# Prefer a project venv if one is present, else whatever python3 is on PATH.
# Only used to read package metadata -- never to import mlx.
PY="python3"
for candidate in "${MLX_VENV_PATH:-}/bin/python" \
                 "${PWD}/.venv/bin/python" \
                 "${PWD}/.venv-mlx/bin/python"; do
  if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
    PY="${candidate}"
    break
  fi
done

notes=()

add_note() { notes+=("$1"); }

# --- chip and NAX -----------------------------------------------------------
hardware=""
if [ -f "${DETECT}" ]; then
  hardware="$("${PY}" "${DETECT}" --text 2>/dev/null)" || hardware=""
fi
[ -z "${hardware}" ] && add_note "Could not determine the chip; detect-apple-silicon.py did not run."

# --- package versions, without importing ------------------------------------
# importlib.metadata reads dist-info from disk. It does not execute the package.
pkg_report="$("${PY}" - <<'PYEOF' 2>/dev/null
from importlib.metadata import version, PackageNotFoundError
out = []
for name in ("mlx", "mlx-lm", "mlx-data", "numpy"):
    try:
        out.append(f"{name} {version(name)}")
    except PackageNotFoundError:
        out.append(f"{name} (not installed)")
print(", ".join(out))
PYEOF
)" || pkg_report=""

if [ -z "${pkg_report}" ]; then
  add_note "Could not read package metadata with ${PY}."
elif printf '%s' "${pkg_report}" | grep -q 'mlx (not installed)'; then
  add_note "mlx is not installed for ${PY}. Install with 'uv pip install mlx mlx-lm' into a Python >= 3.10 environment; MLX requires Apple silicon and macOS >= 14."
fi

# --- Metal toolchain --------------------------------------------------------
# mx.fast.metal_kernel JIT-compiles through the Metal framework and does NOT need
# the metal CLI. Building MLX C++/Metal extensions and capturing GPU traces do.
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  add_note "The 'metal' compiler is unavailable (Command Line Tools only, no full Xcode). Runtime kernels via mx.fast.metal_kernel still work; building MLX C++/Metal extensions and Xcode GPU capture do not."
fi

# --- assemble ---------------------------------------------------------------
{
  printf '%s\n' "${hardware}"
  [ -n "${pkg_report}" ] && printf 'Installed: %s\n' "${pkg_report}"
  if [ "${#notes[@]}" -gt 0 ]; then
    printf '\nEnvironment notes:\n'
    for n in "${notes[@]}"; do printf -- '- %s\n' "${n}"; done
  fi
} | "${PY}" -c '
import json, sys
text = sys.stdin.read().strip()
if text:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        }
    }))
' 2>/dev/null

exit 0
