import numpy as np
import torch
import ddmatch

def get_composition(shape):
    compose_ = ddmatch.core.generate_optimized_diffeo_composition(np.zeros(shape))
    def compose(g1, g2):
        tmp0, tmp1 = np.zeros_like(g1)
        g10, g11 = g1.numpy()
        g20, g21 = g2.numpy()
        compose_(g10, g11, g20, g21, tmp0, tmp1)
        return torch.tensor([tmp0, tmp1])
    return compose

# the following is largely inspired by lie_grp_diffeo

class DiffeoGroup:
    """
    A diffeomorphism group.
    """
    def __init__(self, shape):
        self.shape = shape
        self.composition_ = get_composition(shape)

    def get_raw_identity(self, requires_grad=False):
        """
        Identity diffeomorphisms as tensors.
        """
        idx, idy = np.meshgrid(np.arange(self.shape[0],dtype=float), np.arange(self.shape[1], dtype=float))
        tensor = torch.tensor([idx, idy], requires_grad=requires_grad)
        return tensor

    def element(self, data=None, data_inv=None):
        if data is None and data_inv is None:
            data, data_inv =  [self.get_raw_identity() for i in range(2)]
        elif data is None or data_inv is None:
            raise ValueError()
        return Diffeo(self, data, data_inv)

    def identity(self, requires_grad=True):
        elt = self.element()
        elt.data.requires_grad = requires_grad
        return elt

    def compose_(self, d1, d2):
        return self.composition_(d1, d2)

    def exponential_(self, velocity):
        """
        Approximate exponential by forward (Euler) method.
        """
        return self.get_raw_identity() + velocity

    def exponential(self, velocity):
        """
        An approximation of the exponenttial.
        """
        data = self.exponential_(velocity)
        data_inv = self.exponential_(-velocity)
        return self.element(data, data_inv)


class Diffeo:
    """
    A diffeomorphism and its inverse.
    """
    def __init__(self, group, data, data_inv):
        self.group = group
        self.data = data
        self.data_inv = data_inv

    def compose(self, other):
        data = self.group.compose_(self.data, other.data)
        data_inv = self.group.compose_(other.data_inv, self.data_inv)
        return self.group.element(data, data_inv)

