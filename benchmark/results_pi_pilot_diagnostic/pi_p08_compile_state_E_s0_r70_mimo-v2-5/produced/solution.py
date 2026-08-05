import mlx.core as mx


def make_counter():
    total = mx.array(0.0)

    def _step(x, total):
        return total + mx.sum(x)

    compiled = mx.compile(_step)

    def step(x):
        nonlocal total
        total = compiled(x, total)
        return total

    return step
