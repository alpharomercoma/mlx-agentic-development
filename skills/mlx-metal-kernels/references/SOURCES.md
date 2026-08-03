# Sources

Original condensed writing. Nothing mirrored.

| Page | Upstream |
|---|---|
| `../SKILL.md` | [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and `python/src/fast.cpp` in [ml-explore/mlx](https://github.com/ml-explore/mlx) (MIT) |
| `../SKILL.md` | [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), Table 5.8 attributes |

## Verified by execution

Established by running `../scripts/probe_kernel_binding.py` against MLX 0.32.0 on an
Apple M5, 2026-08-04. **Re-run it rather than trusting the summary below** — MLX ships
roughly every 2.5 weeks and these are properties of the installed version, not
constants.

- Scalar binding by rank: Python scalars and 0-d arrays bind by value and must be
  used bare; 1-d arrays bind as pointers and must be subscripted, even with a single
  element. Both mismatches fail to compile, with different errors.
- `grid` and `threadgroup` are both in threads.
- Output buffers are uninitialised unless `init_value` is passed.
