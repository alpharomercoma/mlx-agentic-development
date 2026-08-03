import mlx.core as mx

MODE = "mxfp4"


def pack_mxfp4(w):
    # mxfp4 returns TWO values, not three: there is no bias term. Code written for
    # affine mode unpacks three and raises.
    wq, scales = mx.quantize(w, mode=MODE)
    return (wq, scales)


def matmul_mxfp4(x, packed):
    wq, scales = packed
    # biases must be omitted for mxfp4; group_size and bits default to the mode's
    # own values (32 and 4), so they are left unspecified rather than copied from
    # affine's defaults of 64 and 4.
    return mx.quantized_matmul(x, wq, scales=scales, transpose=True, mode=MODE)
