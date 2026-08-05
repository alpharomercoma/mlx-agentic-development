import mlx.core as mx
from functools import partial


def make_counter():
    """Return an independent compiled step function that accumulates mx.sum(x)."""

    state = [mx.array(0.0)]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x):
        state[0] = state[0] + mx.sum(x)
        return state[0]

    return step
