import mlx.core as mx


def make_counter() -> callable:
    """Create an independent, compiled running-sum counter."""
    # Keep the state in a mutable container so the compiled function can update
    # the captured array in place through mx.compile's state mechanism.
    state = [mx.array(0.0)]

    def update(x):
        state[0] = state[0] + mx.sum(x)
        return state[0]

    compiled_update = mx.compile(update, inputs=state, outputs=state)

    def step(x):
        return compiled_update(x)

    return step
