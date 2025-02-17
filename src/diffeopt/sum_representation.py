import warnings
import torch
from .group.representation import Representation, Perturbation


class RepresentationProduct(torch.nn.Module):
    """
    A product of representations of one group on
    different vector spaces of the same dimension.
    """
    def __init__(self, *reps: Representation):
        super(RepresentationProduct, self).__init__()
        self.reps = torch.nn.ModuleList(reps)
        # tie perturbations together
        for rep in self.reps[1:]:
            rep.perturbation = self.reps[0].perturbation

    @property
    def perturbation(self) -> Perturbation:
        return self.reps[0].perturbation

    def forward(self, *xs: torch.Tensor) -> list[torch.Tensor]:
        return [rep(x) for rep, x in zip(self.reps, xs)]

def get_sum_representation(rep1: Representation, rep2: Representation) -> RepresentationProduct:
    warnings.warn("Use `RepresentationProduct(rep1, rep2)` instead", DeprecationWarning, stacklevel=2)
    return RepresentationProduct(rep1, rep2)

def OrbitProblem(rep1: Representation, rep2: Representation) -> RepresentationProduct:
    warnings.warn("Use `RepresentationProduct(rep1, rep2)` instead", DeprecationWarning, stacklevel=2)
    return RepresentationProduct(rep1, rep2)
