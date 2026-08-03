"""One-step training helper for MLX models."""

import mlx.core as mx
import mlx.nn as nn


def _mse_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Mean squared error of the model predictions for one batch."""
    return mx.mean(mx.square(model(x) - y))


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array) -> float:
    """Apply one optimizer update for the MSE objective and return its loss."""
    loss, grads = nn.value_and_grad(model, _mse_loss)(model, x, y)
    optimizer.update(model, grads)

    # MLX uses lazy evaluation.  Materialize both the update and its optimizer
    # state so callers observe the new parameters on return.
    mx.eval(model.parameters(), optimizer.state)
    return float(loss)
