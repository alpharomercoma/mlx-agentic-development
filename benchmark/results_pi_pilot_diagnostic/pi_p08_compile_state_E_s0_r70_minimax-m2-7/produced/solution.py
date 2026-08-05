import mlx.core as mx


def make_counter() -> callable:
    """Create an independent compiled counter function."""
    total = 0.0  # Mutable state stored in closure

    def compute_new_total(current_total: float, x: mx.array) -> mx.array:
        """Compiled computation: add sum of x to running total."""
        return current_total + mx.sum(x)

    # Wrap the computation in mx.compile for performance
    compiled_compute = mx.compile(compute_new_total)

    def step(x) -> mx.array:
        """Add mx.sum(x) to running total and return the new total."""
        nonlocal total
        total = compiled_compute(total, x)
        return total

    return step
