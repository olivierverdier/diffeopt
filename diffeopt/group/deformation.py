from dataclasses import dataclass, field
from .base import BaseDiffeoGroup, Diffeo
import torch

@dataclass
class Deformation:
    """
    A container for a big or small deformation.
    """
    group: BaseDiffeoGroup
    _velocity: torch.Tensor = field(init=False)
    deformation: Diffeo = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.velocity = self.group.zero()

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: torch.Tensor) -> None:
        self._velocity = velocity
        self.deformation = self.group.exponential(velocity)
