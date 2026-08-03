`model.py` in this directory defines a small MLX neural network. Create `solution.py`
exposing:

    train_step(model, optimizer, x, y) -> float

It must perform exactly one optimisation step minimising the mean squared error
between `model(x)` and `y`, updating the model's parameters in place, and return the
scalar loss value for this step as a Python float.

`optimizer` is an instance from `mlx.optimizers`.
