# MLX Agentic Development

Skills and agents for Apple MLX development, usable from both Claude Code and OpenAI
Codex. The skill bodies under `skills/` are the single source of truth; `.agents/skills/`
holds symlinks so Codex discovers them at repo scope.

## Using the skills

Invoke by name — `$mlx-metal-kernels` in Codex, `/mlx-metal-kernels` in Claude Code —
or just describe the task and let description matching route you.

| Skill | Use when |
|---|---|
| `mlx-env-setup` | Installing MLX, or `import mlx` fails |
| `mlx-core-semantics` | Lazy evaluation, dtypes, indexing, porting from numpy or PyTorch |
| `mlx-compile-and-transforms` | `mx.compile`, gradients, a loop that runs but does not learn |
| `mlx-metal-kernels` | Writing or debugging a custom GPU kernel |
| `mlx-quantization` | 4-bit and other quantised weights, quantised matmul |
| `mlx-performance` | Making MLX fast; Neural Accelerators; memory and device APIs |
| `mlx-profiling` | Capturing a Metal GPU trace |
| `mlx-lm-workflows` | Running, serving, converting, or fine-tuning models |

Agents: `mlx-kernel-agent` (entry point) and `mlx-performance-agent` (read-only).
Claude Code reads `agents/*.md`; Codex reads the TOML equivalents in `.codex/agents/`.

## Facts that override stale training data

MLX ships roughly monthly and moved substantially through 2026. When working here:

- **MLX is at 0.32.0, mlx-lm at 0.31.3** (2026-08-04), not 0.1x.
- **`mx.metal.*` memory and device functions are deprecated** in favour of top-level
  `mx.*`. The notice prints to **stderr from C++**, not as a Python
  `DeprecationWarning`, so `warnings.catch_warnings` will not see it.
- **M5 has Neural Accelerators, reached implicitly** through `mx.matmul`, `mx.addmm`,
  `mx.quantized_matmul`, and `mx.fast.scaled_dot_product_attention` — never called
  directly. Gated at runtime on macOS ≥ 26.2 **and** GPU architecture generation ≥ 17.
  Checking the chip name alone is wrong.
- **`mx.fast.metal_kernel`'s `grid` is in threads, not threadblocks**, and `source` is
  the kernel body only. Outputs are **uninitialised** unless `init_value` is passed.
- **Scalar kernel inputs bind by rank**: Python scalars and 0-d arrays bind by value
  (`scale`); 1-d arrays bind as pointers (`scale[0]`), even with one element.
- **`mx.quantize` returns three values for `affine` but two for `mxfp4`/`mxfp8`/
  `nvfp4`**, and each mode has its own default group size and bit width.
- **Indexing is not bounds-checked.** Out-of-range indices are undefined behaviour
  returning garbage, not an `IndexError`, because exceptions cannot cross the GPU
  boundary.
- **`mx.compile` freezes captured values at trace time.** Mutating them afterwards is
  silently ignored. Declare state through `inputs=`/`outputs=`.
- **`mx.value_and_grad` differentiates argument 0; `nn.value_and_grad` differentiates
  the model's parameters.** Using the wrong one raises nothing and never updates the
  weights.
- **`mlx-examples` is superseded by the standalone `mlx-lm` repo. `mlx-onnx` is dead.**
  `mlx-vlm`, `mlx-audio`, `mlx-lm-lora` are community packages, not ml-explore.

## Contributing

See `CONTRIBUTING.md`. Two rules matter most: never copy from the Neuron or XLA kits,
and never write an API page from memory — pin a version and verify against it.
