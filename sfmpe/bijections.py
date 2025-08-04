from flax import nnx
from jax import tree
from jaxtyping import PyTree

class Bijection(nnx.Module):
    def __init__(self, forward, inverse):
        super().__init__()
        self.forward = forward
        self.inverse = inverse

    def __call__(self, x):
        return self.forward(x)

    def inverse(self, x):
        return self.inverse(x)

class FittableBijection(Bijection):
    def fit(self, x: PyTree):
        del x
        raise NotImplementedError

class ZScalePyTree(FittableBijection):

    mean: PyTree = 0.
    std: PyTree = 1.

    def __init__(self):
        def forward(x):
            return tree.map(
                lambda leaf, mean, std: (leaf - mean) / std,
                x,
                self.mean,
                self.std
            )
        def inverse(x):
            return tree.map(
                lambda leaf, mean, std: leaf * std + mean,
                x,
                self.mean,
                self.std
            )
        super().__init__(forward, inverse)

    def fit(self, x: PyTree):
        self.mean = tree.map(lambda x: x.mean(0), x)
        self.std = tree.map(lambda x: x.std(0), x)
        print(f'mean: {self.mean}, std: {self.std}')
