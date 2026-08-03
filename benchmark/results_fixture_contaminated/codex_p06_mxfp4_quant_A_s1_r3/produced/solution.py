"""MXFP4 weight packing and quantized matrix multiplication helpers."""

import mlx.core as mx


_MODE = "mxfp4"


def pack_mxfp4(w):
    """Quantize an ``[out_features, in_features]`` weight matrix as MXFP4."""
    weights, scales = mx.quantize(w, mode=_MODE)
    return weights, scales


def matmul_mxfp4(x, packed):
    """Compute ``x @ w.T`` using an MXFP4-packed weight matrix."""
    weights, scales = packed
    return mx.quantized_matmul(
        x, weights, scales=scales, transpose=True, mode=_MODE
    ).astype(mx.float32)
