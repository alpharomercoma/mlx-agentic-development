import mlx.core as mx
import mlx.nn as nn


def train_step(model, optimizer, x, y):
    def loss_fn(m, x, y):
        return mx.mean((m(x) - y) ** 2)

    # nn.value_and_grad, not mx.value_and_grad: it differentiates with respect to
    # model.trainable_parameters() rather than argument 0.
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss.item()
