# MLX Agentic Development

Skills and agents for developing with **Apple MLX** on Apple silicon — custom Metal
kernels, quantisation, M5 Neural Accelerator routing, and performance work — for both
**Claude Code** and **OpenAI Codex**.

And, unusually for a kit of this kind, **a pre-registered experiment measuring whether
it actually helps.**

## Why the experiment

Two predecessors — AWS's `neuron-agentic-development` for Trainium and
`xla-agentic-development` for TPU — assert that a skills kit makes a coding agent
better on unfamiliar accelerator hardware. Neither tested that claim. This repository
does, against both Codex and Claude Code, with a placebo arm, a leakage audit, and
metrics fixed in advance.

See [`benchmark/PREREGISTRATION.md`](benchmark/PREREGISTRATION.md) and
[`benchmark/MINING.md`](benchmark/MINING.md).

**The finding that shaped it:** running bare Codex on ten MLX tasks gave a pass rate
of 9/10 but a token cost varying **8×**, correlating with web-search count at
**r = 0.993**. With documentation available, a strong model facing an unfamiliar
accelerator API usually does not get it wrong — **it pays.** So correctness and
efficiency are co-primary, and efficiency is where the effect is expected.

## Install

**Claude Code**

```
/plugin marketplace add alpharomercoma/mlx-agentic-development
/plugin install mlx-agentic-development@alpharomercoma
```

**Codex** — clone and work inside it; Codex discovers skills from `.agents/skills`.

## Skills

| Skill | Use when |
|---|---|
| [`mlx-env-setup`](skills/mlx-env-setup/SKILL.md) | Installing MLX, or `import mlx` fails |
| [`mlx-core-semantics`](skills/mlx-core-semantics/SKILL.md) | Lazy evaluation, dtypes, indexing, porting from numpy or PyTorch |
| [`mlx-compile-and-transforms`](skills/mlx-compile-and-transforms/SKILL.md) | `mx.compile`, gradients, a loop that runs but does not learn |
| [`mlx-metal-kernels`](skills/mlx-metal-kernels/SKILL.md) | Writing or debugging a custom GPU kernel |
| [`mlx-quantization`](skills/mlx-quantization/SKILL.md) | 4-bit and other quantised weights |
| [`mlx-performance`](skills/mlx-performance/SKILL.md) | Making MLX fast; Neural Accelerators; memory APIs |
| [`mlx-profiling`](skills/mlx-profiling/SKILL.md) | Capturing a Metal GPU trace |
| [`mlx-lm-workflows`](skills/mlx-lm-workflows/SKILL.md) | Running, serving, converting, fine-tuning models |

## What it actually knows

The value is in facts that are easy to get wrong and expensive to rediscover:

- **`mx.fast.metal_kernel`'s `grid` is in threads, not threadblocks** — the CUDA habit
  that silently computes a fraction of your array.
- **Scalar kernel inputs bind by rank.** Python scalars and 0-d arrays bind by value
  (`scale`); 1-d arrays bind as pointers (`scale[0]`), even with one element. Mixing
  the two is the failure that cost bare Codex an entire task.
- **Kernel outputs are uninitialised** unless `init_value` is passed.
- **M5 Neural Accelerators are reached implicitly** via `mx.matmul`, `mx.addmm`,
  `mx.quantized_matmul`, and `mx.fast.scaled_dot_product_attention` — gated on macOS
  ≥ 26.2 **and** GPU architecture generation ≥ 17. The chip name alone is not enough.
- **`mx.quantize` returns three values for `affine` but two for `mxfp4`/`mxfp8`/
  `nvfp4`**, each with its own defaults.
- **Indexing is not bounds-checked** — out-of-range reads are undefined behaviour, not
  an exception.
- **`mx.compile` freezes captured values**, and **`mx.value_and_grad` on a model
  closure silently trains nothing.**
- **`mx.metal.*` memory APIs are deprecated**, and say so on stderr from C++ where
  Python's `warnings` cannot see it.

Chip constants live in
[`hooks/scripts/detect-apple-silicon.py`](hooks/scripts/detect-apple-silicon.py) and
are read at runtime, never hardcoded.

## Verification status

[`VERIFICATION.md`](VERIFICATION.md) records what was executed on real hardware and
what was not, with dates. Profiling and mlx-lm content is documentation-derived and
marked as such.

## Relationship to the Neuron and XLA kits

Structure only. No code or prose was copied; see
[`ATTRIBUTION.md`](ATTRIBUTION.md).

## License

Apache-2.0.
