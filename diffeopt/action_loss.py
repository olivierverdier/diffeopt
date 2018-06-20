import torch

class ActionLoss(torch.nn.Module):
    """
    General loss given
    - an orbit cost C
    - an action
    - a candidate I
        J(h) := C(I.h)
    """
    def __init__(self, action, cost):
        super(ActionLoss, self).__init__()
        self.action = action
        self.cost = cost

    def forward(self, x, g, h):
        self.shifted = self.action(x, g)
        # TODO: is this really necessary?
        perturbed = self.action(self.shifted, h)
        return self.cost(perturbed)
