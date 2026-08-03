# MLX Agentic Development

Skills and agents for developing with **Apple MLX** on Apple silicon — custom Metal
kernels, quantisation, M5 Neural Accelerator routing, and performance work — for both
**Claude Code** and **OpenAI Codex**.

And, unusually for a kit of this kind, **a harness and pre-registered protocol for
measuring whether it actually helps** — with no result yet.

## Why the experiment

Two predecessors — AWS's `neuron-agentic-development` for Trainium and
`xla-agentic-development` for TPU — assert that a skills kit makes a coding agent
better on unfamiliar accelerator hardware. Neither tested that claim. This repository
does, against both Codex and Claude Code, with a placebo arm, a leakage audit, and
metrics fixed in advance.

See [`RESULTS.md`](RESULTS.md),
[`benchmark/PREREGISTRATION.md`](benchmark/PREREGISTRATION.md) and
[`benchmark/MINING.md`](benchmark/MINING.md).

**Result: null.** 250 runs, 5 arms, 44M tokens. **No pre-registered contrast reached
significance.** The sharpest finding is that the full 8-skill kit and a flat 40-line
cheat sheet containing the same facts performed identically — a 3,582-token difference,
p = 0.69 — with the kit loading on 50 of 50 runs. On this task set, with this model,
**the facts did the work and the format did not.** See [`RESULTS.md`](RESULTS.md).

**What the baseline does show:** bare Codex on ten MLX tasks passed **9/10**, with
token cost varying **8×** and correlating with web-search count at **r = 0.993**. With
documentation available, a strong model facing an unfamiliar accelerator API usually
does not get it wrong — **it pays**.

**And what the leakage audit shows, which is worse for the kit:** all **10 of 10**
tasks are in-kit, three at 100% symbol coverage. The kit states the answer to every
task it is tested on. So a kit-versus-bare gain would measure how much cheaper it is
to be *told* than to search — not that the kit makes the agent better. The contrasts
that survive are against a raw-docs arm and a bare-facts arm, where the facts are held
constant.

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

**Facts are re-measured, not frozen.** Each skill ships a probe script that derives
its numbers on your machine at your version: `probe_kernel_binding.py` compiles all six
rank/usage combinations, `measure_quant_error.py` prints the error separation and the
interval a valid tolerance must occupy, `check_deprecations.py` reports what is
actually deprecated now. MLX ships roughly every 2.5 weeks, so a written number has a
shelf life of about one release.

The hook's constants table in
[`hooks/scripts/detect-apple-silicon.py`](hooks/scripts/detect-apple-silicon.py) is
**hardcoded and populated from one machine, an M5** — the hook must not import mlx, so
it cannot ask. On M1–M4 it reports unknown and points at
[`probe_mlx_env.py`](skills/mlx-env-setup/scripts/probe_mlx_env.py), which asks Metal
directly and works on every part.

## Scope — what this does not cover

This is a **kernel-and-inference kit**, not a production MLX kit. It does not cover
`mlx.nn` (71 layer classes), `mlx.optimizers`, training loops, serving depth beyond a
CLI, testing, packaging or Swift shipping, vision-language or audio, `mx.export`, or
distributed execution. `mlx.nn` and `mlx.optimizers` are roughly half the Python
package, and both are absent.

Hardware claims are verified on **one machine, an Apple M5**. `mlx-profiling` and
`mlx-lm-workflows` are documentation-derived and say so at the top of the page —
nothing on those pages was executed.

## Verification status

[`VERIFICATION.md`](VERIFICATION.md) records what was executed on real hardware and
what was not, with dates.

## Relationship to the Neuron and XLA kits

Structure only. No code or prose was copied; see
[`ATTRIBUTION.md`](ATTRIBUTION.md).

## License

Apache-2.0.
