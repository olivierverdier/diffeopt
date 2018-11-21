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

    def zero(self):
        new_shape = [2,] + list(self.shape)
        return torch.zeros(new_shape, dtype=torch.float64)

    def get_raw_identity(self, requires_grad=False):
        """
        Identity diffeomorphisms as tensors.
        """
        idx, idy = np.meshgrid(np.arange(self.shape[0],dtype=float), np.arange(self.shape[1], dtype=float))
        tensor = torch.tensor([idx, idy], requires_grad=requires_grad)
        return tensor

    def element(self, forward=None, backward=None):
        if forward is None and backward is None:
            forward, backward =  [self.get_raw_identity() for i in range(2)]
        # elif forward is None or backward is None:
        #     raise ValueError()
        return Diffeo(self, forward, backward)

    def identity(self, requires_grad=True):
        elt = self.element()
        elt.forward.requires_grad = requires_grad
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
        forward = self.exponential_(velocity)
        backward = self.exponential_(-velocity)
        return self.element(forward, backward)


class Diffeo:
    """
    A diffeomorphism and its inverse.
    """
    def __init__(self, group, forward, backward):
        self.group = group
        self.forward = forward
        self.backward = backward

    def compose(self, other):
        forward = self.group.compose_(self.forward, other.forward)
        backward = self.group.compose_(other.backward, self.backward)
        return self.group.element(forward, backward)

    def inverse(self):
        return self.group.element(self.backward, self.forward)
