"""MXFP4 weight packing and matrix multiplication helpers."""

import mlx.core as mx


def pack_mxfp4(w):
    """Pack a float32 ``[out_features, in_features]`` weight matrix as MXFP4.

    Group size and bit width are deliberately omitted: MLX selects the defaults
    defined by its ``mxfp4`` mode (currently 32 and 4, respectively).
    """
    return mx.quantize(w, mode="mxfp4")


def matmul_mxfp4(x, packed):
    """Compute ``x @ w.T`` directly from weights returned by ``pack_mxfp4``."""
    w_q, scales = packed
    y = mx.quantized_matmul(x, w_q, scales, mode="mxfp4", transpose=True)
    return y.astype(mx.float32)
