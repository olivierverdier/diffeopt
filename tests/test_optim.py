import torch
from diffeopt.group.ddmatch.representation import FunctionRepresentation, DensityRepresentation
from diffeopt.group.ddmatch.group import DiffeoGroup
from diffeopt.sum_representation import get_sum_representation
from diffeopt.distance.information import information_distance
from torch.nn import MSELoss
from diffeopt.cometric.laplace import get_laplace_cometric
from diffeopt.utils import normalize

mse = MSELoss()

from diffeopt.optim import GroupOptimizer

def test_orbit_optimisation():
    shape = (16, 16)
    I0, I1 = [1 + torch.rand(*shape, dtype=torch.float64) for i in range(2)]
    group = DiffeoGroup(I0.shape)
    cometric = get_laplace_cometric(group, s=2)

    sum_rep = get_sum_representation(FunctionRepresentation(group), DensityRepresentation(group))
    oo = GroupOptimizer(sum_rep.parameters(), lr=.1, cometric=cometric)
    vol = normalize(torch.ones_like(I1))
    vol__ = vol + 1e-2*torch.randn_like(vol)
    for i in range(2):
        oo.zero_grad()
        I_, vol_ = sum_rep(I0, vol)
        loss = mse(I_, I0) + information_distance(vol_, vol__)
        loss.backward()
        oo.step()


from torch.nn import Sequential
from diffeopt.optim import VelocityOptimizer

def test_deep_optimisation():
    shape = (16, 16)
    I0, I1 = [1 + torch.rand(*shape, dtype=torch.float64) for i in range(2)]
    group = DiffeoGroup(I0.shape)
    cometric = get_laplace_cometric(group, s=2)

    seq = Sequential(*[FunctionRepresentation(group) for i in range(3)])
    do = VelocityOptimizer(seq.parameters(), lr=.1, cometric=cometric, weight_decay=1.)
    for i in range(2):
        do.zero_grad()
        I_ = seq(I0)
        loss = mse(I_, I1)
        loss.backward()
        do.step()

