---
name: mlx-facts
description: |
  Assorted facts about the MLX array framework on Apple silicon: custom Metal
  kernels, quantisation modes, compilation and function transforms, indexing,
  memory and device APIs, and Neural Accelerator availability. Use when
  writing or debugging MLX code, or when the user mentions "MLX",
  "metal_kernel", "quantize", "mx.compile", "value_and_grad", or "mx.metal".
---

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
