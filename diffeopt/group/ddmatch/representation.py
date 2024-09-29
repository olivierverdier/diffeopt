from typing import Callable, Union
import torch

from ..representation import Representation
from ..base import BaseDiffeoGroup, Diffeo

from .action.function import get_composition_action

class FunctionRepresentation(Representation):
    def get_representation(self, group: BaseDiffeoGroup) -> Callable[[torch.Tensor, Union[torch.Tensor, Diffeo]], torch.Tensor]:
        return get_composition_action(group.shape)

from .action.density import get_density_action

class DensityRepresentation(Representation):
    def get_representation(self, group: BaseDiffeoGroup) -> Callable[[torch.Tensor, Union[torch.Tensor, Diffeo]], torch.Tensor]:
        return get_density_action(group.shape)
