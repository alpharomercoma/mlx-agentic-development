"""4-bit grouped weight packing and matrix multiplication using MLX."""

import mlx.core as mx


_GROUP_SIZE = 64
_BITS = 4


def pack(w):
    """Quantize an ``[out_features, in_features]`` float32 weight matrix.

    MLX returns the packed values together with the per-group scales and biases;
    keep that tuple intact so it can be passed directly to :func:`qmatmul`.
    """
    return mx.quantize(w, group_size=_GROUP_SIZE, bits=_BITS)


def qmatmul(x, packed):
    """Return ``x @ dequantize(packed).T`` using MLX's quantized matmul."""
    weight, scales, biases = packed
    return mx.quantized_matmul(
        x.astype(mx.float32),
        weight,
        scales,
        biases,
        transpose=True,
        group_size=_GROUP_SIZE,
        bits=_BITS,
    )
