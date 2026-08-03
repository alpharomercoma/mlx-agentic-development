# Sources

Original condensed writing. Nothing mirrored.

| Page | Upstream |
|---|---|
| `SKILL.md` | MLX usage docs: [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html), [indexing](https://ml-explore.github.io/mlx/build/html/usage/indexing.html), [numpy interop](https://ml-explore.github.io/mlx/build/html/usage/numpy.html) (MIT) |

## Verified by execution — MLX 0.32.0, Apple M5, 2026-08-04

- Out-of-range gathers return unowned memory rather than raising; the clamp-then-mask
  pattern above was tested against indices far outside the array and is reproducible.
