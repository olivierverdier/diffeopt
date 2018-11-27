import numpy as np
import torch


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
    shape_scale = (group.shape[0] + group.shape[1])/2
    rv_ = rv/nb_steps/shape_scale**2*scale*32
    current = group.element()
    for i in range(nb_steps):
        current = group.exponential(rv_).compose(current)
    return current
