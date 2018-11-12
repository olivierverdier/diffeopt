import torch

class ActionLoss(torch.nn.Module):
    """
    General loss given
    - an orbit cost C
    - an action
    - a candidate I
        J(h) := C(I.h)
    """
    def __init__(self, action, cost, template):
        super(ActionLoss, self).__init__()
        self.action = action
        self.cost = cost
        self.template = template

    def forward(self, g, h):
        self.shifted = self.action(self.template, g)
        # TODO: is this really necessary?
        perturbed = self.action(self.shifted, h)
        return self.cost(perturbed)
