---
name: mlx-compile-and-transforms
description: |
  mx.compile and MLX's function transforms: grad, value_and_grad, vmap, and the
  module-aware nn variants. Use when the user says "mx.compile", "compiled function
  gives stale results", "gradients are zero", "parameters aren't updating",
  "value_and_grad", "nn.value_and_grad", "vmap", "shapeless", or a training loop
  runs without error but does not learn.
---

# Compile and transforms

## Complexity Assessment

**Simple** — one gradient or one compiled function. Read "The two traps" and stop.

**Medium** — a compiled training step with optimiser state. Read that plus
"Compiled training step".

**Complex** — custom vjp, checkpointing, shapeless compile. Read
`references/transforms.md`.

## The two traps

**1. `mx.value_and_grad` differentiates argument 0. `nn.value_and_grad`
differentiates the model's parameters.**

Using the wrong one is the single most damaging mistake in MLX training code,
because it **raises nothing**. The loop runs, the loss is computed, the optimiser
steps, and the weights never change.

```python
# WRONG when you meant to train a model: differentiates whatever is passed first.
loss, grads = mx.value_and_grad(loss_fn)(x, y)

# RIGHT: differentiates model.trainable_parameters(), and respects freeze().
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
```

The symptom is a loss that does not fall. The diagnostic is to snapshot a parameter
before the step and compare after — if nothing changed, this is why.

**2. `mx.compile` freezes anything you did not pass in.**

A compiled function is traced with placeholder arrays. Anything captured from an
enclosing scope is baked in as a **constant at trace time**. Mutating it afterwards
is silently ignored — no error, stale results forever.

```python
from functools import partial

state = [mx.zeros(())]

@partial(mx.compile, inputs=state, outputs=state)   # state declared, not captured
def step(x):
    state[0] = state[0] + mx.sum(x)
    return state[0]
```

Related consequences:
- Compiled functions must be **pure**. Printing or evaluating inside one crashes,
  because the placeholders hold no data. Appending to a list leaves placeholders in
  the list.
- Include `mx.random.state` in the captured state if the function uses randomness,
  or every call draws the same numbers.
- Recompilation is triggered by a change in shape, ndim, dtype, or number of inputs.
  `mx.compile(lambda ...)` inside a loop recompiles every iteration.
- `shapeless=True` avoids shape recompiles but silently bakes in the first shape
  wherever the graph depends on it. Prefer `x.flatten(0, 1)` over
  `x.reshape(x.shape[0] * x.shape[1], -1)`.
- A transform of a compiled function is **not** compiled. Compile the outermost
  function.
- Debug with `mx.disable_compile()` or `MLX_DISABLE_COMPILE=1`. If the bug vanishes,
  it is a compile trap.

## Compiled training step

```python
state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

for x, y in data:
    loss = step(x, y)
    mx.eval(state)
```

## Honesty rails

- **"The loop ran" is not "the model trained."** Verify that parameters changed and
  that the loss actually fell before claiming a training step works.
- If you used `mx.compile` and did not compare against the uncompiled version, say
  so — the compile traps are silent by construction.

## Related

`mlx-core-semantics` for lazy evaluation and `mx.eval` placement.
