from tensorboardX import SummaryWriter
import torch

class Descent:
    """
    Gradient descent on a Lie group based on loss function
    """
    def __init__(self, loss, cometric, group, rate=5e-0):
        """
        loss: Loss function defined on the group.
        cometric: linear map transforming a momentum into a velocity
        group: group with composition, exponential and identity
        rate: learning rate, or time step

        The descent takes a few steps:
        1. compute the loss `loss(current, identity)`
        2. obtain a momentum by taking the derivative wrt the last term
        3. compute the velocity with the cometric
        4. update the group element with the exponential of the velocity
        """
        self.loss = loss
        self.cometric = cometric
        self.rate = rate
        self.group = group

    def initialize(self):
        """
        Initialize writer, current group element and step.
        """
        self.writer = SummaryWriter()
        self.current = self.group.element()
        self.step = 0

    def compute_momentum(self, current):
        """
        Compute momentum at current group element `current`.
        """
        # prepare identity diffeo
        idall = self.group.element().data
        idall_ = torch.tensor(idall, requires_grad=True)
        # compute loss at identity
        current_loss = self.loss(current, idall_)
        # compute momentum
        current_loss.backward()
        momentum = idall_.grad
        # check if something went wrong
        # TODO: check how much the following check slows down each step
        if torch.sum(torch.isnan(momentum)):
            raise Exception("NaN")
        return current_loss, momentum

    def integrate(self, momentum):
        """
        Return a new group element updated from current
        in the direction `velocity`, which is computed
        from the momentum with the cometric.
        """
        # TODO: should be a method of a group class
        # compute velocity
        velocity = self.cometric(momentum)
        # compute exponential
        increment = self.group.exponential(velocity)
        # update by composition with increment
        updated = self.current.compose(increment)
        return updated

    def increment(self):
        current_loss, momentum = self.compute_momentum(self.current)
        self.log(current_loss)
        updated = self.integrate(-self.rate*momentum)
        self.current = updated
        self.step += 1

    def log(self, loss):
        self.writer.add_scalar('loss', loss, self.step)

    def run(self, nb_steps):
        for i in range(nb_steps):
            self.increment()

