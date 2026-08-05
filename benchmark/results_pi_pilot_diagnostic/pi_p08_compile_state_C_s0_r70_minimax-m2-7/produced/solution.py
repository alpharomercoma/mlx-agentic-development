import mlx.core as mx


def make_counter():
    """Return a step(x) function that accumulates running totals across calls."""
    # Mutable state must be declared to mx.compile via inputs/outputs to update.
    total = [mx.array(0.0)]

    def step(x):
        total[0] = total[0] + mx.sum(x)
        return total[0]

    return mx.compile(step, inputs=total, outputs=total)
