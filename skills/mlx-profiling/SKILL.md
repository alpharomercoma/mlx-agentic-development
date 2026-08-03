---
name: mlx-profiling
description: |
  Capture and read GPU traces for MLX on Apple silicon. Use when the user says
  "profile MLX", "Metal trace", "gputrace", "Xcode Instruments", "where is the time
  going", "MTL_CAPTURE_ENABLED", or "mx.metal.start_capture".
---

# Profiling MLX

> **Unverified.** Nothing on this page was executed. Only Command Line Tools are
> installed on the machine this kit was written on, so the Xcode Metal debugger was
> never opened and no trace was ever captured. The API surface and the two failure
> modes below are documentation-derived. Treat them as a starting point, verify before
> relying on them, and prefer the timing recipe in `mlx-performance`, which was
> measured.

## Complexity Assessment

**Start here, and usually stop here.** There is no MLX profiler. Before capturing
anything, check GPU utilisation with a system monitor. If the GPU is not near
saturation, the bottleneck is outside MLX — data loading, host round-trips, or
`.item()` calls forcing partial evaluation — and a GPU trace will not show it.

**Medium** — timing comparisons. Use the benchmarking recipe in `mlx-performance`;
it is the right tool far more often than a trace.

**Complex** — genuine kernel-level investigation. Capture a trace, below.

## Capturing a trace

```python
import mlx.core as mx

mx.metal.start_capture("trace.gputrace")  # path must NOT already exist
...
mx.metal.stop_capture()
```

Two things that silently defeat this:

1. **The process must run with `MTL_CAPTURE_ENABLED=1`.** Without it, capture fails
   quietly and you get nothing.
2. **The PyPI wheel is not built with `MLX_METAL_DEBUG=ON`**, so the trace has no
   shader source and unlabelled command queues. A readable trace needs a source
   build: `CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pip install .`

Open the result in Xcode's Metal debugger. **This requires full Xcode**, not just
Command Line Tools. `mx.fast.metal_kernel` itself does not — it JIT-compiles through
the framework — so kernel *authoring* works on a machine where kernel *profiling*
does not.

`mx.metal.start_capture` / `stop_capture` / `is_available` are **not** deprecated,
unlike the `mx.metal.*` memory functions.

## Honesty rails

- **A trace shows where time went, not why.** State the inference and what would
  falsify it.
- **Duty cycle is not utilisation.** A GPU busy 100% of the time may be doing
  entirely low-intensity work.
- Say whether your MLX build had `MLX_METAL_DEBUG` on. Reading an unlabelled trace
  and reporting confident attributions is guesswork.

## Related

`mlx-performance` for the timing recipe and thermal caveats, which answer most
questions without a trace at all.
