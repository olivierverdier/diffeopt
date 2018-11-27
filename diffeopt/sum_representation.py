import torch

def get_sum_representation(rep1, rep2):
    class OrbitProblem(torch.nn.Module):
        def __init__(self):
            super(OrbitProblem, self).__init__()
            self.rep1 = rep1
            self.rep2 = rep2
            # tie both perturbations together
            self.rep2.perturbation = self.rep1.perturbation

        def forward(self, x1, x2):
            return torch.stack([self.rep1(x1), self.rep2(x2)])
    return OrbitProblem()
