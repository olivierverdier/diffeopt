import torch
from .group.base import BaseDiffeoGroup


def get_volume(shape: tuple) -> torch.Tensor:
    """
    An arbitrary volume form.
    """
    return torch.sum(torch.ones(shape))

def normalize(I: torch.Tensor) -> torch.Tensor:
    """
    Normalize a density
    """
    factor = get_volume(I.shape)/I.sum()
    return I*factor


from diffeopt.cometric import laplace

def get_random_diffeo(group: BaseDiffeoGroup, nb_steps:int=10, scale:float=1., generator: torch.Generator=torch.random.default_generator) -> torch.Tensor:
    cometric = laplace.get_laplace_cometric(group, s=2)
    rm = torch.randn(*group.zero().shape, generator=generator)
    rv = cometric(rm)
    vmx = rv.abs().max()
    shape_scale = (group.shape[0] + group.shape[1])/2
    rv_ = rv/nb_steps/shape_scale**2*scale*32
    current = group.identity()
    for i in range(nb_steps):
        current = group.exponential(rv_).compose(current)
    return current
