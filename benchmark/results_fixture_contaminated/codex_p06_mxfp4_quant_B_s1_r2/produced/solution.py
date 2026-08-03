"""MXFP4 weight packing and matrix multiplication helpers."""

import mlx.core as mx


def pack_mxfp4(w: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize a row-major weight matrix using MLX's mxfp4 defaults."""
    # MXFP4 has no per-group bias, so quantize returns (packed_weights, scales).
    return mx.quantize(w, mode="mxfp4")


def matmul_mxfp4(x: mx.array, packed: tuple[mx.array, mx.array]) -> mx.array:
    """Compute ``x @ w.T`` from weights produced by :func:`pack_mxfp4`."""
    weights, scales = packed
    result = mx.quantized_matmul(x, weights, scales, mode="mxfp4")
    return result.astype(mx.float32)
