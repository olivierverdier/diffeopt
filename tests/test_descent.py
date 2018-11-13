
import numpy as np
import torch

from diffeopt.descent import Descent
from diffeopt.action_loss import ActionLoss
from diffeopt.action.density import get_density_action
from diffeopt.group import DiffeoGroup
from diffeopt.cometric.laplace import get_laplace_cometric
from diffeopt.distance.information import information_distance

def test_run():
    # dist = torch.nn.MSELoss()
    dist = information_distance
    shape = (16, 16)
    I0, I1 = [1 + torch.rand(*shape, dtype=torch.float64) for i in range(2)]
    act = get_density_action(I0.shape)
    def cost(img):
        return dist(img, I1)
    loss = ActionLoss(act, cost, I0)
    group = DiffeoGroup(shape)
    ident = group.element().data
    des = Descent(loss, get_laplace_cometric(ident), group)
    des.initialize()
    for i in range(2):
        des.increment()
