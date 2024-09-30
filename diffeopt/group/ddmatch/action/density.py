from typing import Callable, Any, Union
import torch
import numpy as np
from ddmatch.core import (  # type: ignore
    generate_optimized_image_gradient,
    generate_optimized_image_composition,
    )

from ...base import ForwardDiffeo

# TODO: implement forward or backward in numba??

def get_density_action(shape: torch.Size, compute_id: bool=False) -> Callable[[torch.Tensor, Union[torch.Tensor, ForwardDiffeo]], torch.Tensor]:
    image = np.zeros(shape)
    compute_grad = generate_optimized_image_gradient(image)
    compute_pullback = generate_optimized_image_composition(image)
    class DensityAction(torch.autograd.Function):
        """
        Pullback of x*vol by g and its derivative wrt g.
        """

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor, g: Union[torch.Tensor, ForwardDiffeo]) -> torch.Tensor:
            ctx.save_for_backward(x)
            if isinstance(g, ForwardDiffeo):
                torch_data = g.forward
            else:
                # g must be the identity tensor
                if not compute_id:
                    return x
                else:
                    torch_data = g
            g_ = torch_data.detach().numpy()
            gx_x, gx_y, gy_x, gy_y = [np.zeros(shape) for i in range(4)]
            compute_grad(g_[0], gx_x, gx_y)
            compute_grad(g_[1], gy_x, gy_y)
            gx_x[:,0] += shape[0]/2
            gx_x[:,-1] += shape[0]/2
            gy_y[0,:] += shape[1]/2
            gy_y[-1,:] += shape[1]/2

            jac = gx_x*gy_y - gx_y*gy_x  # Jacobian

            pb = np.zeros(np.shape(x))
            compute_pullback(x.detach().numpy(), g_[0], g_[1], pb)
            res = torch.from_numpy(pb*jac)
            return res

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:  # type: ignore[override]
            f_, = ctx.saved_tensors
            f = f_.detach().numpy()
            p = grad_output.detach().numpy()
            df0, df1 = [np.zeros(shape) for i in range(2)]
            dpf0, dpf1 = [np.zeros(shape) for i in range(2)]
            compute_grad(f, df0, df1)
            compute_grad(f*p, dpf0, dpf1)
            res = torch.tensor(p*np.array([df0, df1]) - np.array([dpf0, dpf1]))
            return (None, res)


        @staticmethod
        def backward_(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
            """
            For the record, here is a theoretically equivalent,
            and simpler version of the momentum map.
            """
            mom = grad_output.detach().numpy()
            resx, resy = [np.zeros(shape) for i in range(2)]
            compute_grad(mom, resx, resy)
            res = torch.tensor([resx, resy])
            out, = ctx.saved_tensors
            return (None, (-res))

    return DensityAction.apply
