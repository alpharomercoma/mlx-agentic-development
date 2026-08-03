# Verification status

What has been checked against reality, and what has not. Nothing here is claimed as
tested that was not.

**Pinned versions:** MLX 0.32.0, mlx-lm 0.31.3. **Machine:** Apple M5, 10-core GPU,
24 GB unified, macOS 26.5.2, `applegpu_g17g`.

## Verified by execution

| Item | How | Date |
|---|---|---|
| NAX availability on this machine | `mx.device_info()` gives `applegpu_g17g`, generation 17; macOS 26.5.2. Both halves of MLX's `is_nax_available()` gate satisfied | 2026-08-04 |
| `detect-apple-silicon.py` gate logic | Tested against macOS below/at/above 26.2, unknown chips, and the `p`-architecture branch requiring generation 18 | 2026-08-04 |
| `check-mlx-env.sh` | Ran with and without MLX installed; warn-only, exit 0, valid `additionalContext` JSON, never imports mlx | 2026-08-04 |
| Metal kernel scalar binding by rank | Direct experiment: Python scalar and 0-d array bind by value, 1-d array binds as pointer; both mismatches fail to compile with distinct errors | 2026-08-04 |
| `mx.quantize` return arity per mode | `affine` 3 values; `mxfp4`, `mxfp8`, `nvfp4` 2 values | 2026-08-04 |
| Quantisation error separations | 4-bit affine: 7.7e-4 correct vs 9.3e-2 unquantised. mxfp4: 4.1e-4 vs 1.2e-1 | 2026-08-04 |
| `mx.fast.scaled_dot_product_attention` numerics and speed | 3.7e-4 relative Frobenius error vs naive float32; 3.59x on [2,8,512,64] | 2026-08-04 |
| `mx.metal.*` deprecation channel | Notices print to stderr from C++; `warnings.catch_warnings` does not observe them | 2026-08-04 |
| matmul throughput | float32 ~1.65 TFLOP/s at 1024², ~9.06 at 2048², ~9.09 at 4096²; bfloat16 ~13.5 at 4096² | 2026-08-04 |
| `mx.compile` captured-state staleness | A counter closing over its accumulator returns stale values and raises nothing | 2026-08-04 |
| Unchecked indexing | Out-of-range gathers return unowned memory; clamp-then-mask verified reproducible | 2026-08-04 |
| `mx.fast.rope` array offsets | Per-sequence array offset matches a per-batch loop of scalar-offset calls to 1e-5 | 2026-08-04 |

## Not verified

| Item | Blocked on |
|---|---|
| Everything in `mlx-profiling` | Full Xcode. Only Command Line Tools are installed, so the `metal` compiler and the Xcode Metal debugger are unavailable. The capture API and its failure modes are documentation-derived |
| Everything in `mlx-lm-workflows` | No model was downloaded or served. The 24 GB sizing figures extrapolate from upstream measurements on a 64 GB M4 Max |
| Building MLX from source, C++/Metal extensions | Full Xcode |
| MLX CUDA backend claims | No CUDA machine |
| Non-M5 chip constants | `detect-apple-silicon.py` deliberately leaves M1–M4 unpopulated rather than guessing their Metal architecture strings, since the NAX gate reads that string and a wrong value would mis-report accelerator availability |
| Whether NAX is *engaged* for a given op | The gate is confirmed and throughput is consistent with it, but MLX exposes no runtime switch to A/B the NAX path, so engagement per op is inferred rather than proven |
| Claude Code and Codex plugin install end to end | Not yet installed from the marketplace in either client |

## Coverage, stated as a limit

The kit covers custom kernels, quantisation, core array semantics, compilation and
transforms, performance routing, and mlx-lm basics. It does **not** cover `mlx.nn`
(71 layer classes), `mlx.optimizers` (including `Muon` and `MultiOptimizer`), training
loops, `mx.export_function`, serving internals, testing, packaging or Swift shipping,
vision-language or audio, or `mx.distributed`. `mlx.nn` and `mlx.optimizers` are
roughly half the Python package.

Every hardware claim is from **one machine**: an Apple M5, 10-core GPU, 24 GB, macOS
26.5.2. Nothing is verified on M1–M4, on a Pro/Max/Ultra part, or on the Linux/CUDA
backend.

## Superseded by probe scripts

The measured constants below are retained for the record, but the authority has moved
to the probe scripts, which re-derive them on the running machine at the installed
version. Prefer running the script to trusting the table; MLX ships roughly every 2.5
weeks and several of these figures are known to have moved upstream since.

| Claim | Now derived by |
|---|---|
| NAX gate, chip architecture | `skills/mlx-env-setup/scripts/probe_mlx_env.py` |
| quantize return arity, defaults | `skills/mlx-quantization/scripts/probe_quant_modes.py` |
| quantisation error tolerances | `skills/mlx-quantization/scripts/measure_quant_error.py` |
| kernel scalar binding by rank | `skills/mlx-metal-kernels/scripts/probe_kernel_binding.py` |
| `mx.metal.*` deprecation set | `skills/mlx-performance/scripts/check_deprecations.py` |

## How to move an item up

Run it, record the result here with a date. A documented failure is worth more than
an untested claim.
