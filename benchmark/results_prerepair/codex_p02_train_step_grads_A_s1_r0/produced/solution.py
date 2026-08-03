"""Single-step training helper for MLX models."""

import mlx.core as mx
import mlx.nn as nn


def _mse_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Mean squared error for the supplied model and batch."""
    return mx.mean((model(x) - y) ** 2)


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array) -> float:
    """Optimise ``model`` once on ``(x, y)`` and return its MSE loss."""
    loss_and_grad = nn.value_and_grad(model, _mse_loss)
    loss, gradients = loss_and_grad(model, x, y)
    optimizer.update(model, gradients)

    # MLX operations are lazy; evaluate before converting to a Python scalar.
    mx.eval(loss)
    return float(loss.item())
