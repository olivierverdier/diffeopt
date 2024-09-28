import warnings
import torch
from .group.representation import Representation


class OrbitProblem(torch.nn.Module):
    def __init__(self, rep1: Representation, rep2: Representation):
        super(OrbitProblem, self).__init__()
        self.rep1 = rep1
        self.rep2 = rep2
        # tie both perturbations together
        self.rep2.perturbation = self.rep1.perturbation

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.rep1(x1), self.rep2(x2)])

def get_sum_representation(rep1: Representation, rep2: Representation) -> OrbitProblem:
    warnings.warn("Use `OrbitProblem(rep1, rep2)` instead", DeprecationWarning, stacklevel=2)
    return OrbitProblem(rep1, rep2)
