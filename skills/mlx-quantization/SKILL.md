---
name: mlx-quantization
description: |
  Quantise weights and run quantised matmuls in MLX. Use when the user says
  "quantize", "4-bit", "8-bit", "mxfp4", "nvfp4", "mxfp8", "mx.quantize",
  "quantized_matmul", "QuantizedLinear", "nn.quantize", "shrink this model",
  "convert to MLX 4-bit", or hits an unpacking or shape error from quantisation.
---

# Quantisation in MLX

## Complexity Assessment

**Simple** — quantise a whole model. `nn.quantize(model, group_size=64, bits=4)`
mutates it in place. Done.

**Medium** — hand-rolled pack/matmul, or a non-affine mode. Read the mode table
below; it is the thing that breaks people.

**Complex** — mixed precision per layer, activation quantisation, learned
quantisation (DWQ/AWQ/GPTQ). Run `scripts/probe_quant_modes.py` to print the actual
return arity and defaults for every mode in your installed version.

## The mode table — read this before writing any code

`mx.quantize` **does not return the same number of values in every mode.**

| Mode | Returns | Default group_size | Default bits | Scale type |
|---|---|---|---|---|
| `affine` (default) | **3**: `w_q, scales, biases` | 64 | 4 | input dtype |
| `mxfp4` | **2**: `w_q, scales` | 32 | 4 | e8m0 |
| `mxfp8` | **2**: `w_q, scales` | 32 | 8 | e8m0 |
| `nvfp4` | **2**: `w_q, scales` | 16 | 4 | e4m3 |

Verified against MLX 0.32.0. Affine supports **2, 3, 4, 5, 6, and 8 bits** — not just
4 and 8. **Run `scripts/probe_quant_modes.py` rather than trusting this table**: it
prints the return arity, effective group size, and dequantise dtype for every mode in
the installed version. It also shows a detail the table above understates — `dequantize`
defaults to float32 for `affine` but **bfloat16** for the other three modes.

Consequences:

```python
# Affine: three values, and biases must be passed on.
w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
y = mx.quantized_matmul(
    x, w_q, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
)

# mxfp4: TWO values. Unpacking three raises. biases must be omitted, and the
# defaults differ, so do not copy affine's group_size=64.
w_q, scales = mx.quantize(w, mode="mxfp4")
y = mx.quantized_matmul(x, w_q, scales=scales, transpose=True, mode="mxfp4")
```

## Other sharp edges

- `transpose=True` is the **default**, and it means `w` is `[out, in]` so the product
  is `x @ w.T`. That is usually what you want for a `Linear` weight.
- The last dimension of `w` must be divisible by `group_size`.
- `mx.dequantize(..., dtype=None)` **infers the dtype and defaults to bfloat16**, not
  to the original float32. Pass `dtype=mx.float32` explicitly when you need it.
- `nn.quantize` mutates the model **in place** and returns None. It quantises every
  leaf exposing `to_quantized()` — `nn.Linear` and `nn.Embedding` by default. Pass
  `class_predicate` to control it; the predicate may return a dict of kwargs for
  mixed precision.
- Activation quantisation (`quantize_input=True`) works only for `nvfp4` and `mxfp8`,
  and only on `Linear`.

## Verifying a quantised implementation

`mx.quantized_matmul` does **not** equal dequantise-then-matmul bit for bit; it
accumulates differently. Measured on M5, MLX 0.32.0, relative Frobenius error:

| Comparison | 4-bit affine | mxfp4 |
|---|---|---|
| `quantized_matmul` vs dequantise-then-matmul | 7.7e-4 | 4.1e-4 |
| full-precision matmul vs dequantise-then-matmul | 9.3e-2 | 1.2e-1 |

So a correct implementation sits two orders of magnitude closer to the dequantised
reference than an unquantised one. **Run `scripts/measure_quant_error.py`** to get
these numbers for your machine and version — it prints the separation and the interval
any valid tolerance must sit inside. Upstream moves these figures; a commit fixing
nvfp4 through the split-K path landed within a month of the version above.

If your quantised result matches full precision *too* well, you probably are not
quantising at all.

## Honesty rails

- **State the mode, bits, and group size** behind any number you report. "4-bit" is
  ambiguous across modes.
- **A tolerance chosen so the test passes is not a verification.** Say how you picked
  it.
- Quantisation trades accuracy for memory. Do not report a memory win without saying
  what happened to accuracy, or that you did not measure it.

## Related

`mlx-performance` — `mx.quantized_matmul` is a NAX-accelerated path on M5.
