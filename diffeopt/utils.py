import numpy as np
import torch
import ddmatch

def get_identity(shape, requires_grad=False):
    """
    Identity diffeomorphisms.
    """
    idx, idy = np.meshgrid(np.arange(shape[0],dtype=float), np.arange(shape[1], dtype=float))
    tensor = torch.tensor([idx, idy], requires_grad=requires_grad)
    return tensor

def get_exponential(shape):
    id_ = get_identity(shape)
    def exponential(velocity):
        """
        Approximate exponential by forward (Euler) method.
        """
        return id_ + velocity
    return exponential

def get_composition(shape):
    compose_ = ddmatch.core.generate_optimized_diffeo_composition(np.zeros(shape))
    def compose(g1, g2):
        tmp0, tmp1 = np.zeros_like(g1)
        g10, g11 = g1.numpy()
        g20, g21 = g2.numpy()
        compose_(g10, g11, g20, g21, tmp0, tmp1)
        return torch.tensor([tmp0, tmp1])
    return compose

def get_distance_to(dist, target):
    """
    Utility function: given a distance and a target,
    return the cost function of the distance to that target.
    """
    def matching(img):
        return dist(img, target)
    return matching

def get_volume(shape):
    """
    An arbitrary volume form.
    """
    return np.sum(np.ones(shape))

def normalize(I):
    """
    Normalize a density
    """
    factor = get_volume(I.shape)/I.sum()
    return I*factor
