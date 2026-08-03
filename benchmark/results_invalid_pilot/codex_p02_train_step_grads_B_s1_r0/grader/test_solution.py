import os
import sys

sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))
import mlx.core as mx
import mlx.optimizers as optim
import solution
from mlx.utils import tree_flatten
from model import MLP


def _params(m):
    return {k: mx.array(v) for k, v in tree_flatten(m.parameters())}


def test_returns_float_loss():
    mx.random.seed(0)
    m = MLP()
    opt = optim.SGD(learning_rate=1e-2)
    x = mx.random.normal((16, 8))
    y = mx.random.normal((16, 1))
    mx.eval(m.parameters(), x, y)
    loss = solution.train_step(m, opt, x, y)
    assert isinstance(loss, float), f"expected float, got {type(loss)}"
    assert loss == loss, "loss is NaN"


def test_parameters_actually_change():
    # The central check. Differentiating the wrong argument leaves parameters
    # untouched and raises nothing.
    mx.random.seed(0)
    m = MLP()
    opt = optim.SGD(learning_rate=1e-2)
    x = mx.random.normal((16, 8))
    y = mx.random.normal((16, 1))
    mx.eval(m.parameters(), x, y)
    before = _params(m)
    solution.train_step(m, opt, x, y)
    mx.eval(m.parameters())
    after = _params(m)
    changed = [k for k in before if not mx.allclose(before[k], after[k], atol=0).item()]
    assert changed, "no parameter changed; the optimiser step did not reach the weights"


def test_loss_decreases_over_steps():
    mx.random.seed(0)
    m = MLP()
    opt = optim.SGD(learning_rate=5e-2)
    x = mx.random.normal((64, 8))
    y = mx.random.normal((64, 1))
    mx.eval(m.parameters(), x, y)
    first = solution.train_step(m, opt, x, y)
    for _ in range(60):
        last = solution.train_step(m, opt, x, y)
    assert last < first * 0.9, f"loss did not fall: {first:.5f} -> {last:.5f}"


def test_loss_matches_mse():
    mx.random.seed(1)
    m = MLP()
    opt = optim.SGD(learning_rate=0.0)  # no-op step
    x = mx.random.normal((8, 8))
    y = mx.random.normal((8, 1))
    mx.eval(m.parameters(), x, y)
    expected = mx.mean((m(x) - y) ** 2).item()
    got = solution.train_step(m, opt, x, y)
    assert abs(got - expected) < 1e-4, f"reported loss {got} != MSE {expected}"
