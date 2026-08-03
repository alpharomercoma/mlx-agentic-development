# Sources

Original condensed writing. Nothing mirrored.

| Page | Upstream |
|---|---|
| `SKILL.md` | [MLX compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) and [function transforms](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html) docs; `python/mlx/nn/utils.py` in [ml-explore/mlx](https://github.com/ml-explore/mlx) (MIT) |

## Verified by execution — MLX 0.32.0, Apple M5, 2026-08-04

- A counter closing over its accumulator under `mx.compile` returns stale values and
  raises nothing; declaring the same state via `inputs=`/`outputs=` fixes it.
