"""A compiled, stateful MLX counter."""

from functools import partial

import mlx.core as mx


def make_counter():
    """Return an independent function that accumulates sums of its inputs."""
    # A container lets ``mx.compile`` capture this as both an implicit input and
    # output, so its array is refreshed after every compiled invocation.
    state = [mx.array(0)]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x):
        state[0] = state[0] + mx.sum(x)
        return state[0]

    return step
