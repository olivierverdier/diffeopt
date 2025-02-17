from typing import Any, Callable, Union, Optional
import torch
import numpy as np

from ...base import Diffeo

from ddmatch.core import generate_optimized_image_gradient, generate_optimized_image_composition  # type: ignore

from ..group import DiffeoGroup

def get_composition_action(shape: torch.Size, compute_id:bool=False) -> Callable[[torch.Tensor, Union[torch.Tensor, Diffeo]], torch.Tensor]:
    """
    compute_id: right composition with the identity
    is the identity map, but it takes time to compute.
    Assuming that the gradient is only required in that case,
    we use the `requires_grad` property to check whether
    the group element is the identity.
    """
    I0 = np.zeros(shape)
    image_compose = generate_optimized_image_composition(I0)
    image_gradient = generate_optimized_image_gradient(I0)

    from .density import get_density_action
    density_action = get_density_action(shape)

    group = DiffeoGroup(shape)

    class CompositionAction(torch.autograd.Function):
        """
        Right composition action
            q(x) --> q(g(x))
        and its derivative wrt g.
        """
        @staticmethod
        def forward(ctx: Any, q: torch.Tensor, g: Union[torch.Tensor, Diffeo]) -> torch.Tensor:
            if isinstance(g, Diffeo):
                torch_data = g.forward
                to_save = g.backward
            else:
                # g must be the identity tensor
                torch_data = g
                to_save = g
            ctx.save_for_backward(q, to_save)
            if not isinstance(g, Diffeo) and not compute_id:
                # if g is a tensor, it must be the identity
                return q
            y = np.zeros(q.shape)
            g_ = torch_data.detach().numpy()
            q_ = q.detach().numpy()
            image_compose(q_, g_[0], g_[1], y)
            res = torch.from_numpy(y)
            return res

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:  # type: ignore[override]
            """
            This is the adjoint of the derivative only
            if g was the identity.
            """
            q, data_inv = ctx.saved_tensors
            if ctx.needs_input_grad[1]:
                # Derivative wrt g = Id
                q_ = q.detach().numpy()
                gxout, gyout = np.zeros_like(q_), np.zeros_like(q_)
                image_gradient(q_, gxout, gyout)
                grad = torch.tensor(np.array([gxout, gyout]))
                result = grad*grad_output
                return (grad_output, result)
            else:
                # Derivative wrt q
                g = group.half_element(data_inv)
                img_grad = density_action(grad_output, g)
                return (img_grad, None)
    return CompositionAction.apply
