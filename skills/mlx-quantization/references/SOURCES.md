# Sources

Original condensed writing. Nothing mirrored.

| Page | Upstream |
|---|---|
| `SKILL.md` | `python/src/ops.cpp` and `python/mlx/nn/layers/quantized.py` in [ml-explore/mlx](https://github.com/ml-explore/mlx) (MIT); [MLX quantization docs](https://ml-explore.github.io/mlx/build/html/python/ops.html) |
| Learned quantisation | [mlx-lm LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) (MIT) |

## Verified by execution — MLX 0.32.0, Apple M5, 2026-08-04

- Return arity by mode: `affine` 3 values, `mxfp4`/`mxfp8`/`nvfp4` 2 values.
- Relative Frobenius errors quoted in the table were measured, not estimated.
