import ddmatch
import numpy as np
import torch
from .base import BaseDiffeoGroup

def get_composition(shape):
    compose_ = ddmatch.core.generate_optimized_diffeo_composition(np.zeros(shape))
    def compose(g1, g2):
        tmp0, tmp1 = np.zeros_like(g1)
        g10, g11 = g1.numpy()
        g20, g21 = g2.numpy()
        compose_(g10, g11, g20, g21, tmp0, tmp1)
        return torch.tensor([tmp0, tmp1])
    return compose

class DiffeoGroup(BaseDiffeoGroup):
    def __init__(self, shape):
        super(DiffeoGroup, self).__init__(shape)
        self.composition_ = get_composition(shape)

    def compose_(self, d1, d2):
        return self.composition_(d1, d2)
