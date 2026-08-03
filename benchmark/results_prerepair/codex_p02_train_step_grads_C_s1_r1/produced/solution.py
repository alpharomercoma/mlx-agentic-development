"""One-step training utility for the MLP model."""

import mlx.core as mx
import mlx.nn as nn


def _mse_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Mean squared prediction error for ``model`` on one batch."""
    return mx.mean((model(x) - y) ** 2)


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array) -> float:
    """Update ``model`` once using MSE and return the pre-update loss."""
    loss, grads = nn.value_and_grad(model, _mse_loss)(model, x, y)
    optimizer.update(model, grads)

    # MLX is lazy: materialise the updated parameters before reporting completion.
    mx.eval(model.parameters())
    return float(loss)
