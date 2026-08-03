"""One-step training helper for the MLX model in :mod:`model`."""

import mlx.core as mx
import mlx.nn as nn


def train_step(model, optimizer, x, y) -> float:
    """Update ``model`` once using mean squared error on ``(x, y)``.

    MLX evaluation is lazy, so the updated parameters and loss are evaluated
    before converting the loss to a host-side Python ``float``.
    """

    def loss_fn(current_model, inputs, targets):
        return mx.mean((current_model(inputs) - targets) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    return float(loss)
