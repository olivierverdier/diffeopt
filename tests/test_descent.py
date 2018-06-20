
import numpy as np
import torch

from diffeopt.descent import Descent
from diffeopt.action_loss import ActionLoss
from diffeopt.action.density import get_density_action
from diffeopt.utils import get_exponential, get_identity, get_composition
from diffeopt.cometric.laplace import get_laplace_cometric
from diffeopt.distance.information import information_distance

def test_run():
    # dist = torch.nn.MSELoss()
    dist = information_distance
    I0, I1 = [1 + torch.rand(16, 16, dtype=torch.float64) for i in range(2)]
    act = get_density_action(I0.shape)
    def cost(img):
        return dist(img, I1)
    loss = ActionLoss(act, cost)
    des = Descent(I0, loss, get_laplace_cometric(I0.shape), get_composition(I0.shape), get_exponential(I0.shape), get_identity(I0.shape))
    des.initialize()
    for i in range(2):
        des.increment()
