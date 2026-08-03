---
name: mlx-performance
description: |
  Make MLX fast on Apple silicon: routing work to the M5 Neural Accelerators, using
  the fused mx.fast operations, and reading memory and device state with the current
  API. Use when the user says "MLX is slow", "speed up MLX", "make this faster on
  M-series", "Neural Accelerator", "NAX", "how much memory is MLX using", "mx.metal
  is deprecated", "peak memory", "wired limit", or asks whether to write a custom
  kernel.
---

# MLX performance on Apple silicon

## Complexity Assessment

**Quick win** — the user has a slow model or op. Check the fused-op table below and
the dtype. Usually finishes here. Read nothing else.

**Measured** — a real regression or an unexplained cost. Follow "Measure before
optimising", then run
`scripts/check_deprecations.py` to confirm which memory APIs are current in your
installed version.

**Deep** — kernel-level work. Run `skills/mlx-env-setup/scripts/probe_mlx_env.py` to confirm Neural
Accelerator eligibility on this machine, then hand off to
`mlx-metal-kernels`.

## Reach for a fused op before writing anything

`mx.fast` ships hand-tuned implementations, and on M5 these are the paths that reach
the Neural Accelerators. A hand-written kernel almost never beats them.

| Instead of | Use |
|---|---|
| manual softmax attention with a mask | `mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask="causal")` |
| manual RMS norm | `mx.fast.rms_norm` |
| manual layer norm | `mx.fast.layer_norm` |
| hand-rolled rotary embedding | `mx.fast.rope` |
| dequantise then matmul | `mx.quantized_matmul` |

Notes that cost people time:
- `scaled_dot_product_attention`'s `mask="causal"` is **lower-right aligned**: the
  last query attends to the last key. That is what you want with a KV cache, and it
  differs from top-left causal when the query and key lengths differ.
- Do **not** pre-tile K/V for grouped-query attention; pass the smaller head count.
- Softmax is done in float32 internally regardless of input dtype, so upcasting
  first is wasted work. The same applies to `rms_norm` and `layer_norm`.
- A fused op will not match a naive implementation bit for bit — it accumulates in a
  different order. Measured on M5: `scaled_dot_product_attention` differs from a
  naive float32 implementation by ~4e-4 relative Frobenius error. That is expected,
  not a bug.

## Neural Accelerators (NAX) on M5

MLX 0.30.0 added kernels targeting the per-core Neural Accelerators. **You never call
them directly.** They are selected automatically inside `mx.matmul`, `mx.addmm`,
`mx.quantized_matmul`, and `mx.fast.scaled_dot_product_attention` when the shapes
qualify. The way to "use NAX" is to route work through those ops.

Availability is gated at runtime on **both** conditions:

- macOS ≥ 26.2, **and**
- GPU architecture generation ≥ 17 (≥ 18 when the architecture string ends in `p`)

Checking the chip name alone is wrong: an M5 on macOS 26.0 has no NAX. Run
`hooks/scripts/detect-apple-silicon.py` rather than assuming; it evaluates both
halves of the real gate.

NAX is young and has had **correctness** bugs, not just performance ones — integer
overflows on KV sequences beyond 32K and a kernel-name mismatch, fixed across
0.30.5–0.32.0. Pin a recent MLX, and check any long-context or quantized result
against a `stream=mx.cpu` reference before trusting it.

## Memory and device state — use the current API

These moved from `mx.metal.*` to top level. The old spellings still work but print a
deprecation notice **to stderr from C++**, not as a Python `DeprecationWarning` — so
`warnings.catch_warnings` will not see them, and they will not show up in a test that
only inspects warnings.

| Use | Not |
|---|---|
| `mx.get_active_memory()` | `mx.metal.get_active_memory()` |
| `mx.get_peak_memory()` / `mx.reset_peak_memory()` | `mx.metal.*` |
| `mx.get_cache_memory()` / `mx.clear_cache()` | `mx.metal.*` |
| `mx.set_memory_limit()` / `mx.set_cache_limit()` | `mx.metal.*` |
| `mx.set_wired_limit()` | `mx.metal.set_wired_limit()` |
| `mx.device_info()` | `mx.metal.device_info()` |

`mx.metal.is_available()` and `mx.metal.start_capture()`/`stop_capture()` are **not**
deprecated.

**The table above is a snapshot; the set moves every release.** Run
`scripts/check_deprecations.py` to print what is actually deprecated in the installed
version, and what each moved to.

`get_active_memory()` excludes cached buffers, so it will not match Activity Monitor.
The default memory limit is 1.5× the recommended working-set size, which means MLX
will let you into swap before it raises.

## Measure before optimising

```python
import time, mlx.core as mx


def bench(fn, *args, n=50, warmup=5):
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / n
```

`mx.eval` must be inside the timed region — MLX is lazy, and timing without it
measures graph construction. `mx.synchronize()` before starting and before stopping.

**On a fanless Mac, thermal drift is a real confound.** Interleave the variants you
are comparing instead of running all of A then all of B, take the minimum of several
interleaved rounds, and report the coefficient of variation. A "speedup" measured
while the chassis was cool is not a speedup.

Small matmuls are launch-bound, not compute-bound. Measured once on a 10-core M5:
float32 matmul reached ~1.7 TFLOP/s at 1024² but ~9.1 TFLOP/s at 2048². **Those are
this machine on one afternoon, not constants** — re-measure with the recipe above
before quoting a number. The shape of the effect is the durable part; the figures are
not.

## Honesty rails

- **Peak figures are vendor peaks, not attainable rates.** Use them for ratios.
- **A measurement is not a conclusion.** State the number, the inference it supports,
  and what would falsify it.
- **Never report a predicted speedup as a measured one.** If you changed code and did
  not re-run the benchmark, say so.
- Report the shapes, dtypes, and repeat count behind any number you quote.

## Related

`mlx-metal-kernels` when a fused op genuinely does not exist.
`mlx-quantization` for reducing memory rather than time.
