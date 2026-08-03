# Sources

Original condensed writing. Nothing mirrored.

| Page | Upstream |
|---|---|
| `SKILL.md`, `kernel-api.md` | [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and `python/src/fast.cpp` in [ml-explore/mlx](https://github.com/ml-explore/mlx) (MIT) |
| `kernel-api.md` | [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), Table 5.8 attributes |
| `traps.md` | Established by execution against MLX 0.32.0 on an Apple M5, 2026-08-04 — see `VERIFICATION.md` |

## Verified by execution

- Scalar binding by rank: Python scalars and 0-d arrays bind by value and must be
  used bare; 1-d arrays bind as pointers and must be subscripted, even with a single
  element. Both mismatches fail to compile, with different errors.
- `grid` and `threadgroup` are both in threads.
- Output buffers are uninitialised unless `init_value` is passed.
