from abc import ABC, abstractmethod
from typing import Callable, Union
from .deformation import Deformation
import torch
from torch.nn.parameter import Parameter
from .base import BaseDiffeoGroup, Diffeo


class Perturbation(Parameter):
    """
    Thin layer on a Parameter class that contains a Deformation object,
    which contains a tensor to be modified in place during
    optimisation.
    """
    deformation: Deformation

    @property
    def base(self) -> Deformation:
        return self.deformation

class Representation(torch.nn.Module, ABC):
    """
    Group representation.
    The parameter is the group element stored in `perturbation`,
    which is an instance of `Parameter`.
    """

    def __init__(self, group: BaseDiffeoGroup, requires_grad:bool=True):
        super(Representation, self).__init__()
        self.representation = self.get_representation(group)
        self.perturbation = Perturbation(group.get_raw_identity(), requires_grad)
        self.perturbation.deformation = Deformation(group)

    def reset_parameters(self) -> None:
        """
        Reset the underlying base point to the group identity.
        """
        self.perturbation.base.reset()

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        return self.representation(self.representation(I, self.perturbation.base.deformation), self.perturbation)

    @abstractmethod
    def get_representation(self, group: BaseDiffeoGroup) -> Callable[[torch.Tensor, Union[torch.Tensor, Diffeo]], torch.Tensor]:
        pass
