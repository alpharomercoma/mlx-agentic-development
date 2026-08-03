"""One-step MSE training utility for MLX models."""

import mlx.core as mx
import mlx.nn as nn


def _mse_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Return the mean squared error for ``model`` on one batch."""
    return mx.mean((model(x) - y) ** 2)


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array) -> float:
    """Update ``model`` once using MSE loss and return that step's loss."""
    loss, grads = nn.value_and_grad(model, _mse_loss)(model, x, y)
    optimizer.update(model, grads)
    return float(loss)
