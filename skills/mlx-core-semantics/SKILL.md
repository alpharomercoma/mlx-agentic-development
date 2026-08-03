---
name: mlx-core-semantics
description: |
  MLX's array semantics: lazy evaluation, unified memory and streams, dtype
  promotion, and unchecked indexing. Use when the user says "mx.eval", "why is my
  MLX result wrong", "MLX memory keeps growing", "dtype changed unexpectedly",
  "index out of bounds", "MLX vs numpy", "to(device)", "MLX is slow in a loop", or
  is porting numpy or PyTorch code to MLX.
---

# MLX core semantics

## Complexity Assessment

**Simple** — one confusing result or a port of a small function. Read "The four
rules" and stop.

**Medium** — a training loop, memory growth, or a numpy port. Add
`references/porting.md`.

**Complex** — custom transforms, streams, mixed-device pipelines. Read everything.

## The four rules

**1. Everything is lazy.** Operations record a graph; nothing computes until you
force it. Evaluation is forced by `mx.eval(...)`, `print`, `.item()`, conversion to
numpy, `mx.save*`, or using a scalar in Python control flow (`if y > 0:`).

Evaluate **once per loop iteration**, not after every operation:

```python
for batch in dataset:
    loss, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)   # once, at the end
```

Forgetting `optimizer.state` is a real bug and not an obvious one: the optimiser's
momentum graph is never evaluated, so it grows every step. It shows up as steadily
increasing memory and slowdown, never as an error.

Calling `.item()` mid-iteration forces a **partial** evaluation — the forward pass
runs, then the backward pass runs separately. Collect losses as arrays and evaluate
them together.

**2. Unified memory: there is no `.to(device)`.** Arrays live in shared memory. You
choose the device **per operation**, not per array: `mx.add(a, b, stream=mx.cpu)`.
Cross-stream dependencies are inserted automatically. `mx.float64` is **CPU-only**
and raises on the GPU — a common surprise when porting numpy.

**3. Dtype promotion is weak for Python scalars.** `x * 2.0` on a bfloat16 array
stays bfloat16. `x * mx.array(2.0)` promotes to **float32**. Wrapping a scalar in
`mx.array` is the quiet way to double your memory and lose your speed while the
numbers still look right. Defaults are float32 for floats, int32 for ints, and
`bfloat16 * float16` promotes to float32 rather than to either.

**4. Indexing is not bounds-checked.** Out-of-range indices are **undefined
behaviour**, not an `IndexError`, because exceptions cannot propagate from the GPU.
You get whatever was in memory. Guard explicitly:

```python
valid = (idx >= 0) & (idx < n)
safe = mx.where(valid, idx, mx.zeros_like(idx))     # clamp BEFORE gathering
out = mx.where(valid, mx.take(x, safe), fill)       # then mask the result
```

Also: slicing **copies** (unlike numpy views), so mutating `b = a[:]` leaves `a`
unchanged; and scatter with duplicate indices is **nondeterministic**.

## Porting reflexes

| PyTorch / numpy | MLX |
|---|---|
| `t.to("mps")`, `.cuda()` | nothing — pass `stream=` per op if you care |
| `loss.backward()`, `zero_grad()`, `detach()` | `mx.value_and_grad` / `nn.value_and_grad`; `mx.stop_gradient` |
| `nn.Parameter` | any public `mx.array` attribute; leading `_` makes it not a parameter |
| `np.nonzero`, boolean *reads* | unsupported — data-dependent shapes have no GPU form |
| `float64` | silently unavailable on GPU; use float32 |

## Honesty rails

- **A number that looks plausible is not a verified number.** When porting, compare
  against the original implementation on real inputs and report the error.
- If you did not run it, say you did not run it. Lazy evaluation makes it unusually
  easy to write MLX code that never actually executed during your testing.

## Related

`mlx-compile-and-transforms` for gradients and `mx.compile`.
`mlx-performance` for memory APIs and measurement.
