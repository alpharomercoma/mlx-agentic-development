"""Single-step training helper for MLX models."""

import mlx.core as mx
import mlx.nn as nn


def _mse_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Return the mean-squared prediction error for a batch."""
    return mx.mean(mx.square(model(x) - y))


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array) -> float:
    """Optimise ``model`` once against ``(x, y)`` and return its MSE loss."""
    loss, grads = nn.value_and_grad(model, _mse_loss)(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss)
