# Sources

Original condensed writing. Nothing mirrored.

| Page | Upstream |
|---|---|
| `SKILL.md` | [MLX docs](https://ml-explore.github.io/mlx/build/html/index.html) (MIT), `python/src/fast.cpp`, `python/src/memory.cpp`, `python/src/metal.cpp` |
| NAX gate | `mlx/backend/metal/device.cpp::is_nax_available()` in [ml-explore/mlx](https://github.com/ml-explore/mlx) (MIT) |
| M5 figures | [Apple M5 announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) |

## Verified by execution on an Apple M5, MLX 0.32.0, 2026-08-04

- `applegpu_g17g`, architecture generation 17, macOS 26.5.2 satisfies the NAX gate.
- Deprecation notices for `mx.metal.*` print to stderr from C++, not through Python
  `warnings`; `warnings.catch_warnings` does not observe them.
- float32 matmul ~1.65 TFLOP/s at 1024², ~9.06 at 2048², ~9.09 at 4096²;
  bfloat16 ~13.5 TFLOP/s at 4096².
- `mx.fast.scaled_dot_product_attention` differs from a naive float32 implementation
  by ~3.7e-4 relative Frobenius error, and delivers ~3.6x on [2,8,512,64].
