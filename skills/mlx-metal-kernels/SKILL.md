---
name: mlx-metal-kernels
description: |
  Write, debug, and tune custom GPU kernels for Apple silicon with
  mx.fast.metal_kernel. Use when the user says "write a Metal kernel", "custom MLX
  kernel", "mx.fast.metal_kernel", "fuse these ops on the GPU", "my MLX kernel won't
  compile", "kernel returns garbage", "speed this MLX op up with a kernel", or needs
  a fused elementwise/reduction operation MLX does not provide.
---

# Custom Metal kernels in MLX

`mx.fast.metal_kernel` compiles Metal at runtime through the Metal framework. It does
**not** need Xcode or the `metal` CLI. (Building MLX C++ extensions and capturing GPU
traces do.)

## Complexity Assessment

Classify first; do not read every reference.

**Simple** — elementwise, one output, one input plus scalars.
Copy the skeleton below, adjust the body, write it. Read nothing else.
Target: working kernel in minutes.

**Medium** — reductions, multiple outputs, threadgroup cooperation, non-float32
dtypes. Read `references/kernel-api.md`. Prefer a threadgroup tree reduction over
atomics.

**Complex** — differentiable kernels, atomics, templates, non-contiguous inputs,
strided access. Read `references/kernel-api.md` and `references/traps.md` in full.

**Before writing any kernel, ask whether one is needed.** `mx.fast` already ships
fused `rms_norm`, `layer_norm`, `rope`, and `scaled_dot_product_attention`, and on
M5 those reach the Neural Accelerator paths that a hand-written kernel will not.
Beating them by hand is unlikely. See `mlx-performance`.

## The skeleton

```python
import mlx.core as mx

# Build ONCE at module scope. Every construction creates and JIT-compiles a new
# Metal library; building inside the hot function dominates runtime.
_kernel = mx.fast.metal_kernel(
    name="my_op",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i < n_elements) {
            out[i] = a[i] + b[i];
        }
    """,
)

def my_op(a, b):
    (out,) = _kernel(
        inputs=[a, b],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(a.size, 1, 1),        # THREADS, not threadgroups
        threadgroup=(256, 1, 1),
    )
    return out
```

## The five things that go wrong

**1. `source` is the body only.** The `[[kernel]] void ...(...)` signature is
generated for you from `input_names`, `output_names`, and `template`. Writing your
own signature is a compile error.

**2. `grid` is in THREADS, not threadblocks.** This is the CUDA habit that breaks
here. `grid=(n, 1, 1)` launches `n` threads total. It is passed to
`dispatchThreads`, so `grid=(ceil(n/256), 1, 1)` launches 256× too few threads and
silently computes only the first fraction of your array. Each `threadgroup`
dimension must not exceed the corresponding `grid` dimension.

**3. Scalars bind by value or by pointer depending on rank.** Verified on
MLX 0.32.0:

| Passed in `inputs` | Bound as | Use in `source` |
|---|---|---|
| Python scalar, e.g. `3.0` | value | `scale` |
| 0-d array, `mx.array(3.0)` | value | `scale` |
| 1-d array, `mx.array([3.0])` — even one element | pointer | `scale[0]` |

Mismatching gives two different compile errors:
`subscripted value is not an array, pointer, or vector` when you subscript a
by-value binding, and `invalid operands to binary expression` when you use a
pointer as a value. **Passing a 0-d array and then writing `scale[0]` is the single
most common failure here.**

**4. Outputs are uninitialised.** `metal_kernel` does not zero its output buffers.
Any element your kernel does not write keeps whatever was in that memory. If the
kernel accumulates (`out[i] += ...`) or writes conditionally, pass `init_value=0`.
Symptoms: correct on the first call, wrong on later ones; results that change
between identical calls; nonzero output for an all-zero input.

**5. Bounds are not checked anywhere.** Neither MLX indexing nor your kernel body
gets bounds checking, because exceptions cannot propagate from the GPU. Always guard
with `if (i < n)`. An out-of-range read returns unowned memory rather than raising.

## Debugging

Reach for these in order.

1. **`verbose=True` on the call** prints the full generated kernel, signature
   included. Most compile errors are obvious once you can see what was actually
   built. This is the first move, not the last.
2. **Compare against a CPU reference**: run the equivalent MLX expression with
   `stream=mx.cpu` and diff. Report the relative Frobenius error, not just
   `allclose`.
3. **Feed an all-zeros input.** The correct answer is exactly zero, so any leftover
   buffer contents are unmissable. This isolates trap 4 immediately.
4. **Call twice with the same input** and compare. Drift means uninitialised output
   or a race.
5. **Shrink the grid to one thread** and check a single element by hand.

## Honesty rails

- **A kernel that runs is not a kernel that is correct.** Say which shapes and dtypes
  you actually tested, including sizes that are not a multiple of the threadgroup.
- **State the tolerance and why.** A tolerance chosen so the test passes is not a
  verification. Measure the error of a known-correct implementation first, then set
  the threshold with margin.
- **Do not claim a speedup you have not measured**, and do not compare against a
  strawman baseline. `mx.fast.*` is the baseline that matters.
- Tail elements are where kernels fail. If you did not test a size that is not a
  multiple of your threadgroup width, say so.

## Related

`mlx-performance` for whether a kernel is the right answer at all.
`mlx-core-semantics` for lazy evaluation and dtype rules that affect kernel inputs.
