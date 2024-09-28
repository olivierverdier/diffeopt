from abc import ABC, abstractmethod
from typing import Callable
from .deformation import Deformation
import torch
from torch.nn.parameter import Parameter
from .base import BaseDiffeoGroup


class Representation(torch.nn.Module, ABC):
    """
    Group representation.
    """

    def __init__(self, group, requires_grad:bool=True):
        super(Representation, self).__init__()
        self.representation = self.get_representation(group)
        self.perturbation = Parameter(group.get_raw_identity(), requires_grad)
        self.perturbation.base = Deformation(group)

    def reset_parameters(self):
        """
        Reset the underlying base point to the group identity.
        """
        self.perturbation.base.reset()

    def forward(self, I):
        return self.representation(self.representation(I, self.perturbation.base.deformation), self.perturbation)

    @abstractmethod
    def get_representation(self, group: BaseDiffeoGroup) -> Callable:
        pass
