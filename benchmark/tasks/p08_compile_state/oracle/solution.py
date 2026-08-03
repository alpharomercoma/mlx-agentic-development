from functools import partial

import mlx.core as mx


def make_counter():
    # State must be passed through inputs= and outputs= rather than captured. A
    # value merely closed over is frozen as a constant when the function is traced,
    # and later mutation of it is silently ignored.
    state = [mx.zeros((), dtype=mx.float32)]

    @partial(mx.compile, inputs=state, outputs=state)
    def _step(x):
        state[0] = state[0] + mx.sum(x)
        return state[0]

    def step(x):
        out = _step(x)
        mx.eval(state, out)
        return out

    return step
