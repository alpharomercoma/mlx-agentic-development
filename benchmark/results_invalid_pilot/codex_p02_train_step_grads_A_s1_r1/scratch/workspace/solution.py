"""One-step training utility for MLX models."""

import mlx.core as mx
import mlx.nn as nn


def train_step(model, optimizer, x, y) -> float:
    """Update ``model`` once to minimise mean squared error on ``(x, y)``."""

    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    mx.eval(loss)
    return float(loss)
