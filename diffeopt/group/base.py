from abc import ABC, abstractmethod
import torch

from dataclasses import dataclass

from typing_extensions import Self



# the following is largely inspired by lie_grp_diffeo

@dataclass
class BaseDiffeoGroup(ABC):
    """
    A diffeomorphism group.
    """

    shape: torch.Size

    def zero(self) -> torch.Tensor:
        new_shape = [2,] + list(self.shape)
        return torch.zeros(new_shape, dtype=torch.float64)

    def get_raw_identity(self, requires_grad:bool=False) -> torch.Tensor:
        """
        Identity diffeomorphisms as tensors.
        """
        idx, idy = torch.meshgrid(torch.arange(self.shape[0],dtype=torch.float64), torch.arange(self.shape[1], dtype=torch.float64), indexing='xy')
        tensor = torch.stack([idx, idy]).requires_grad_(requires_grad)
        return tensor

    def half_element(self, forward: torch.Tensor) -> "ForwardDiffeo":
        return ForwardDiffeo(self, forward)

    def element(self, forward: torch.Tensor, backward: torch.Tensor) -> "Diffeo":
        # forward, backward =  [self.get_raw_identity() for i in range(2)]
        return Diffeo(self, forward, backward)

    def identity(self) -> "Diffeo":
        return self.element(self.get_raw_identity(), self.get_raw_identity())


    @abstractmethod
    def compose_(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        pass

    def exponential_(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Approximate exponential by forward (Euler) method.
        """
        return self.get_raw_identity() + velocity

    def exponential(self, velocity: torch.Tensor) -> "Diffeo":
        """
        An approximation of the exponenttial.
        """
        forward = self.exponential_(velocity)
        backward = self.exponential_(-velocity)
        return self.element(forward, backward)

@dataclass
class ForwardDiffeo:
    group: BaseDiffeoGroup
    forward: torch.Tensor

@dataclass
class Diffeo(ForwardDiffeo):
    """
    A diffeomorphism and its inverse.
    """
    backward: torch.Tensor

    def compose(self, other: Self) -> "Diffeo":
        forward = self.group.compose_(self.forward, other.forward)
        backward = self.group.compose_(other.backward, self.backward)
        return self.group.element(forward, backward)

    def inverse(self: Self) -> "Diffeo":
        return self.group.element(self.backward, self.forward)
