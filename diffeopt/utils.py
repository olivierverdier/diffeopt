import numpy as np
import torch
import ddmatch


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
