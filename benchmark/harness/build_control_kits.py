#!/usr/bin/env python3
"""Generate the arm D and arm E control kits.

Both are delivered through the same mechanism as the real kit -- a `skills/` tree
copied into a per-arm CODEX_HOME -- so the delivery path is held constant and only
the content varies.

**Arm D, raw docs.** Upstream MLX documentation, concatenated and unrouted. The text
is the installed package's own docstrings, so it is genuinely upstream prose rather
than something reworded, and it regenerates against whatever MLX is installed.
Tests whether *curation* beats *concatenation*.

**Arm E, bare facts.** The same load-bearing facts the real kit carries, as a flat
cheat sheet. No per-topic split, no Complexity Assessment, no routing tables, no
honesty rails, no worked examples -- just the claims. Tests whether the skill
*format* contributes anything beyond the text.

Arm E matters most. Because both C and E contain the facts, the leakage that makes
C-A uninterpretable cancels between them, so C-E is the contrast that survives the
corrected leakage audit.

    python3 build_control_kits.py --out benchmark
"""

from __future__ import annotations

import argparse
import inspect
import textwrap
from pathlib import Path

# The API surface the tasks actually touch. Arm D gets these docstrings verbatim.
SYMBOLS = [
    "mx.fast.metal_kernel", "mx.fast.scaled_dot_product_attention", "mx.fast.rope",
    "mx.fast.rms_norm", "mx.fast.layer_norm",
    "mx.quantize", "mx.dequantize", "mx.quantized_matmul",
    "mx.compile", "mx.eval", "mx.value_and_grad", "mx.grad", "mx.vmap",
    "mx.take", "mx.where", "mx.matmul", "mx.addmm", "mx.synchronize",
    "mx.get_active_memory", "mx.get_peak_memory", "mx.get_cache_memory",
    "mx.clear_cache", "mx.set_memory_limit", "mx.set_cache_limit",
    "mx.set_wired_limit", "mx.device_info", "mx.metal.is_available",
    "mx.metal.start_capture", "mx.metal.stop_capture",
    "nn.value_and_grad", "nn.quantize", "nn.average_gradients",
]

# Arm E: the same facts the kit carries, stripped of all structure.
BARE_FACTS = """\
MLX facts

- mx.fast.metal_kernel: `source` is the kernel BODY only; the signature is generated.
- mx.fast.metal_kernel: `grid` and `threadgroup` are in THREADS, not threadblocks.
  grid=(n,1,1) launches n threads. It is passed to dispatchThreads.
- mx.fast.metal_kernel: outputs are UNINITIALISED unless init_value= is passed.
  An accumulating or conditionally-written kernel reads whatever was in memory.
- mx.fast.metal_kernel: scalar inputs bind by RANK. A Python scalar or a 0-d array
  binds by value, used as `scale`. A 1-d array binds as a pointer, used as
  `scale[0]`, even with one element. Mismatching either way fails to compile.
- mx.fast.metal_kernel: constructing one JIT-compiles a Metal library. Build it at
  module scope, not inside the hot function.
- mx.fast.metal_kernel: compile_options={"math_mode": "safe"} preserves IEEE
  behaviour, which -inf masks depend on.
- MLX does not bounds-check indexing. Out-of-range indices are undefined behaviour
  returning unowned memory, not an IndexError, because exceptions cannot propagate
  from the GPU. Clamp before gathering, then mask the result.
- Slicing copies in MLX; it is not a view. Scatter with duplicate indices is
  nondeterministic.
- mx.compile freezes anything not passed through inputs=/outputs= as a constant at
  trace time. Mutating captured state afterwards is silently ignored.
- mx.compile: include mx.random.state in the captured state if the function uses
  randomness. Recompiles on change of shape, ndim, dtype, or number of inputs.
- mx.value_and_grad differentiates argument 0. nn.value_and_grad(model, fn)
  differentiates model.trainable_parameters(). Using the wrong one raises nothing
  and never updates the weights.
- Training loops must mx.eval(model.parameters(), optimizer.state). Omitting
  optimizer.state leaves its graph unevaluated and memory grows every step.
- Python scalars are weakly typed: x * 2.0 keeps bfloat16, x * mx.array(2.0)
  promotes to float32. mx.float64 is CPU-only and raises on the GPU.
- mx.quantize returns THREE values for mode="affine" and TWO for mxfp4, mxfp8 and
  nvfp4. Default group_size/bits: affine 64/4, mxfp4 32/4, mxfp8 32/8, nvfp4 16/4.
  Affine supports 2, 3, 4, 5, 6 and 8 bits.
- mx.quantized_matmul defaults to transpose=True, meaning w is [out, in] and the
  product is x @ w.T. The last dim of w must be divisible by group_size.
- mx.dequantize(dtype=None) infers: float32 for affine, bfloat16 for the others.
- mx.fast.scaled_dot_product_attention mask="causal" is LOWER-RIGHT aligned. Do not
  pre-tile k/v for grouped-query attention. Softmax is done in float32 internally.
- mx.fast.rope accepts an ARRAY offset for per-sequence positions.
- The mx.metal.* memory and device functions are deprecated in favour of top-level
  mx.*; the notice prints to stderr from C++, not through Python warnings, so
  warnings.catch_warnings cannot see it. mx.metal.is_available and start_capture
  are not deprecated.
- Apple M5 has Neural Accelerators, reached implicitly through mx.matmul, mx.addmm,
  mx.quantized_matmul and mx.fast.scaled_dot_product_attention -- never called
  directly. Gated at runtime on macOS >= 26.2 AND GPU architecture generation >= 17
  (>= 18 when the architecture string ends in 'p').
- A fused op does not match a naive implementation bit for bit; it accumulates in a
  different order.
- mx.metal.start_capture needs MTL_CAPTURE_ENABLED=1 or it fails silently, and a
  readable trace needs an MLX_METAL_DEBUG=ON source build.
"""


def resolve(dotted: str):
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

    obj = {"mx": mx, "nn": nn}[dotted.split(".")[0]]
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def build_arm_d(out: Path) -> int:
    d = out / "docs-kit" / "skills" / "mlx-docs"
    d.mkdir(parents=True, exist_ok=True)
    chunks = []
    for name in SYMBOLS:
        obj = resolve(name)
        if obj is None:
            continue
        doc = inspect.getdoc(obj)
        if not doc:
            continue
        chunks.append(f"## {name}\n\n```\n{doc.strip()}\n```")

    body = "\n\n".join(chunks)
    text = (
        "---\n"
        "name: mlx-docs\n"
        "description: |\n"
        "  Upstream MLX API documentation for arrays, custom Metal kernels, fast fused\n"
        "  operations, quantisation, compilation, function transforms, and memory and\n"
        "  device management. Use when writing or debugging MLX code on Apple silicon,\n"
        '  or when the user mentions "MLX", "mx.fast", "metal_kernel", "quantize",\n'
        '  "mx.compile", "value_and_grad", or "mx.metal".\n'
        "---\n\n"
        "# MLX API documentation\n\n"
        "Docstrings from the installed MLX, concatenated in no particular order.\n\n"
        + body
        + "\n"
    )
    (d / "SKILL.md").write_text(text)
    return len(text)


def build_arm_e(out: Path) -> int:
    d = out / "facts-kit" / "skills" / "mlx-facts"
    d.mkdir(parents=True, exist_ok=True)
    text = (
        "---\n"
        "name: mlx-facts\n"
        "description: |\n"
        "  Assorted facts about the MLX array framework on Apple silicon: custom Metal\n"
        "  kernels, quantisation modes, compilation and function transforms, indexing,\n"
        "  memory and device APIs, and Neural Accelerator availability. Use when\n"
        '  writing or debugging MLX code, or when the user mentions "MLX",\n'
        '  "metal_kernel", "quantize", "mx.compile", "value_and_grad", or "mx.metal".\n'
        "---\n\n" + BARE_FACTS
    )
    (d / "SKILL.md").write_text(text)
    return len(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    nd = build_arm_d(args.out)
    ne = build_arm_e(args.out)
    print(f"arm D (raw docs)   {nd:6,} chars -> {args.out}/docs-kit")
    print(f"arm E (bare facts) {ne:6,} chars -> {args.out}/facts-kit")
    print(
        textwrap.dedent("""
        Arm E is deliberately smaller than the real kit. It is not a length control --
        arm B is. It holds the FACTS constant and removes the structure, so C-E
        isolates what the skill format contributes.
        """).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
