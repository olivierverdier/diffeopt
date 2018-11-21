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


from diffeopt.cometric import laplace
def get_random_diffeo(group, nb_steps=10, scale=1.):
    cometric = laplace.get_laplace_cometric(group, s=2)
    rm = np.random.randn(*group.zero().shape)
    rv = cometric(rm)
    vmx = rv.abs().max()
    rv_ = 2*rv/vmx/nb_steps*scale
    current = group.element()
    for i in range(nb_steps):
        current = group.exponential(rv_).compose(current)
    return current
