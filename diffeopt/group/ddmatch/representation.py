from ..representation import Representation

from .action.function import get_composition_action

class FunctionRepresentation(Representation):
    def get_representation(self, group):
        return get_composition_action(group.shape)

from .action.density import get_density_action

class DensityRepresentation(Representation):
    def get_representation(self, group):
        return get_density_action(group.shape)
