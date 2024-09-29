from typing import Callable
from dataclasses import dataclass

import ddmatch  # type: ignore
import numpy as np
import torch
from ..base import BaseDiffeoGroup

def get_composition(shape: torch.Size) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    compose_ = ddmatch.core.generate_optimized_diffeo_composition(np.zeros(shape))
    def compose(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        tmp0, tmp1 = np.zeros_like(g1)
        g10, g11 = g1.numpy()
        g20, g21 = g2.numpy()
        compose_(g10, g11, g20, g21, tmp0, tmp1)
        return torch.from_numpy(np.array([tmp0, tmp1]))
    return compose

@dataclass
class DiffeoGroup(BaseDiffeoGroup):
    def __post_init__(self) -> None:
        self.composition_ = get_composition(self.shape)

    def compose_(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        return self.composition_(d1, d2)


