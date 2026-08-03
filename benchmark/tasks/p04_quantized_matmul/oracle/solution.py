import mlx.core as mx

BITS, GROUP = 4, 64


def pack(w):
    # affine mode returns three values; the mxfp4/mxfp8/nvfp4 modes return two.
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
    return (wq, scales, biases)


def qmatmul(x, packed):
    wq, scales, biases = packed
    # transpose=True (the default) means w is [out, in] and the product is x @ w.T,
    # which is exactly what is wanted here.
    return mx.quantized_matmul(
        x,
        wq,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=GROUP,
        bits=BITS,
    )
