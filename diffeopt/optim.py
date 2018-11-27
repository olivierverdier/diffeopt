import torch

class DiffeoOptimizer(torch.optim.Optimizer):
    def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            for group in self.param_groups:
                cometric = group['cometric']
                for p in group['params']:
                    momentum = p.grad
                    velocity = cometric(momentum)
                    self._update_parameter(p, velocity, group)

    def _update_parameter(self, parameter, velocity, group):
        raise NotImplementedError()


class OrbitOptimizer(DiffeoOptimizer):
    def __init__(self, params, lr, cometric):
        defaults = {'lr': lr, 'cometric': cometric}
        super(OrbitOptimizer, self).__init__(params, defaults)

    def _update_parameter(self, parameter, velocity, group):
        grad_direction = -group['lr']*velocity
        parameter.base.deformation = parameter.base.deformation.compose(parameter.base.group.exponential(grad_direction))


class DeepOptimizer(DiffeoOptimizer):
    def __init__(self, params, lr, cometric, weight_decay):
        defaults = {'lr': lr, 'cometric': cometric, 'weight_decay':weight_decay}
        super(DeepOptimizer, self).__init__(params, defaults)

    def _update_parameter(self, parameter, velocity, group):
        direction = -group['lr']*(velocity + group['weight_decay']*parameter.base.velocity)
        parameter.base.velocity += direction
