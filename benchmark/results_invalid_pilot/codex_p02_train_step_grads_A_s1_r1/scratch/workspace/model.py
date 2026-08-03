import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int = 8, d_hidden: int = 32, d_out: int = 1):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)

    def __call__(self, x: mx.array) -> mx.array:
        return self.l2(nn.relu(self.l1(x)))
