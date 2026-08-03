"""A single optimisation step for the model defined in :mod:`model`."""

import mlx.core as mx
import mlx.nn as nn


def _mse_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Return the mean squared prediction error for one batch."""
    return mx.mean((model(x) - y) ** 2)


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array) -> float:
    """Update ``model`` once on ``(x, y)`` and return that step's MSE loss."""
    loss, grads = nn.value_and_grad(model, _mse_loss)(model, x, y)
    optimizer.update(model, grads)

    # MLX is lazy, so materialise the parameter and optimiser-state updates before
    # handing the scalar result back to the caller.
    mx.eval(loss, model.parameters(), optimizer.state)
    return float(loss)
