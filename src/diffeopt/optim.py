from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
import torch
from .group.representation import Perturbation

from torch.optim import Optimizer  # type: ignore[attr-defined]

class DiffeoOptimizer(Optimizer, ABC):

    @torch.no_grad()
    def step(self, closure:Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cometric = group['cometric']
            for p in group['params']:
                momentum = p.grad
                velocity = cometric(momentum)
                self._update_parameter(p, velocity, group)

        return loss

    @abstractmethod
    def _update_parameter(self, parameter: Perturbation, velocity: torch.Tensor, group: dict[str, Any]) -> None:
        pass


class GroupOptimizer(DiffeoOptimizer):
    """
    Update group elements from a momentum.
    """
    def __init__(self, params: list[torch.nn.Parameter], lr: float, cometric: Callable[[torch.Tensor], torch.Tensor]):
        defaults = {'lr': lr, 'cometric': cometric}
        super(GroupOptimizer, self).__init__(params, defaults)

    def _update_parameter(self, parameter: Perturbation, velocity: torch.Tensor, group: dict[str, Any]) -> None:
        grad_direction = -group['lr']*velocity
        parameter.base.deformation = parameter.base.deformation.compose(parameter.base.group.exponential(grad_direction))


class VelocityOptimizer(DiffeoOptimizer):
    """
    Update velocity from a momentum and a weight decay.
    """
    def __init__(self, params: list[torch.nn.Parameter], lr: float, cometric: Callable[[torch.Tensor], torch.Tensor], weight_decay: float):
        defaults = {'lr': lr, 'cometric': cometric, 'weight_decay':weight_decay}
        super(VelocityOptimizer, self).__init__(params, defaults)

    def _update_parameter(self, parameter: Perturbation, velocity: torch.Tensor, group: dict[str, Any]) -> None:
        direction = -group['lr']*(velocity + group['weight_decay']*parameter.base.velocity)
        parameter.base.velocity += direction
