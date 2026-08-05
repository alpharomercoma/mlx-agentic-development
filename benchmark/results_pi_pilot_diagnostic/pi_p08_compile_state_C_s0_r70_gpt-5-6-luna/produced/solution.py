from functools import partial

import mlx.core as mx


def make_counter():
    """Create an independent, compiled running-sum counter."""
    state = [mx.array(0.0)]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x):
        state[0] = state[0] + mx.sum(x)
        return state[0]

    return step
