"""A small, independently stateful compiled MLX counter."""

import mlx.core as mx


def make_counter() -> callable:
    """Return a function that accumulates the sums of its inputs."""
    total = mx.array(0)

    @mx.compile
    def add_to_total(current_total, value):
        return current_total + mx.sum(value)

    def step(x):
        nonlocal total
        total = add_to_total(total, x)
        return total

    return step
