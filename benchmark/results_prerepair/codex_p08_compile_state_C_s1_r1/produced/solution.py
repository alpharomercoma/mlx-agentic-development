"""A compiled, stateful MLX counter."""

from functools import partial

import mlx.core as mx


def make_counter():
    """Return an independent function that accumulates sums of its inputs."""
    # State must be declared to ``mx.compile``: captured values are otherwise
    # treated as constants from the first trace.
    state = [mx.zeros(())]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x):
        state[0] = state[0] + mx.sum(x)
        return state[0]

    return step
