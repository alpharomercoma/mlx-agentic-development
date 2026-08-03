---
name: mlx-env-setup
description: |
  Install and verify an MLX environment on Apple silicon. Use when the user says
  "install MLX", "pip install mlx", "MLX won't import", "no GPU found", "which
  Python for MLX", "mlx-lm install", "MLX requirements", or a script fails at
  `import mlx`.
---

# MLX environment setup

## Complexity Assessment

**Simple** — fresh install. Run the block below, then the verification gate. Stop.

**Medium** — an existing environment misbehaving. Run the gate first; it usually
names the problem.

**Complex** — building from source, CUDA, or C++ extensions. Read
`references/building.md`.

## Install

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mlx mlx-lm
```

Requirements, all hard:

- **Apple silicon.** MLX's Metal backend does not exist on Intel Macs.
- **Native arm64 Python ≥ 3.10.** A Rosetta Python fails; check with
  `python -c "import platform; print(platform.processor())"` — it must print `arm`.
- **macOS ≥ 14.0** for MLX at all; **≥ 26.2** for the M5 Neural Accelerator paths.

Do not use the system Python on macOS: it is typically 3.9, below MLX's minimum.

Building from source additionally needs full Xcode (≥ 15) and, on M5,
`MACOSX_DEPLOYMENT_TARGET=26.2` — without it you get a binary with no NAX kernels
and silently lose the accelerated paths. The PyPI wheel already includes them.

## Verification gate — run this before anything else

```python
import mlx.core as mx
print(mx.__version__, mx.metal.is_available())
print(mx.device_info())
```

`metal.is_available()` must be True and `device_info()["architecture"]` should look
like `applegpu_gNNx`. If Metal is unavailable, nothing else in the kit applies.

For chip generation, memory, and Neural Accelerator eligibility, run
`hooks/scripts/detect-apple-silicon.py`, which evaluates MLX's actual runtime gate
rather than guessing from the chip name.

## Current versions and dead ends

Pin versions; MLX ships roughly monthly and the API moves.

- `mlx` 0.32.0 and `mlx-lm` 0.31.3 as of 2026-08-04. mlx-lm lags mlx.
- **`mlx-examples` is superseded**: LLM tooling now lives in the standalone
  `ml-explore/mlx-lm` repo and package.
- **`mlx-onnx` is dead** (last commit Feb 2024). Do not use it.
- `mlx-vlm`, `mlx-audio`, and `mlx-lm-lora` are **community** packages, not
  ml-explore. The official Swift-side LLM/VLM home is `ml-explore/mlx-swift-lm`.
- MLX has a **CUDA backend** on Linux (`pip install mlx[cuda12]`), with incomplete
  parity. NAX, Metal capture, and `.metallib` extensions are Metal-only;
  `mx.fast.cuda_kernel` is CUDA-only.

## Honesty rails

- Report the exact versions you verified against. "Works with MLX" is not a claim.
- If you could not run something, say so rather than asserting it works.
