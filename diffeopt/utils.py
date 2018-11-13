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
