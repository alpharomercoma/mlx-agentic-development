"""MXFP4 weight packing and matrix multiplication helpers."""

import mlx.core as mx


def pack_mxfp4(w: mx.array) -> object:
    """Pack an ``[out_features, in_features]`` float32 weight matrix as MXFP4.

    Leaving ``group_size`` and ``bits`` unspecified deliberately selects the
    defaults defined by MLX for the ``mxfp4`` mode.
    """
    return mx.quantize(w, mode="mxfp4")


def matmul_mxfp4(x: mx.array, packed: object) -> mx.array:
    """Compute ``x @ w.T`` from weights returned by :func:`pack_mxfp4`."""
    weights, scales = packed
    result = mx.quantized_matmul(x, weights, scales, mode="mxfp4")
    return result.astype(mx.float32)
