import mlx.core as mx
import mlx.nn as nn


def train_step(model, optimizer, x, y) -> float:
    """Run one MSE optimisation step and return its loss."""

    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return float(loss)
