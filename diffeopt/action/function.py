import torch
import numpy as np

from ddmatch.core import generate_optimized_image_composition, generate_optimized_image_gradient, generate_optimized_diffeo_composition

def get_composition_action(shape, compute_id=False):
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
    diffeo_compose = generate_optimized_diffeo_composition(I0)

    class CompositionAction(torch.autograd.Function):
        """
        Right composition action
            q(x) --> q(g(x))
        and its derivative wrt g.
        """
        @staticmethod
        def forward(ctx, q, g):
            ctx.save_for_backward(q,)
            if g.requires_grad and not compute_id:
                # gradient can only be obtained if g is the identity
                return q
            y = np.zeros(q.shape)
            g_ = g.detach().numpy()
            q_ = q.detach().numpy()
            image_compose(q_, g_[0], g_[1], y)
            res = torch.from_numpy(y)
            return res

        @staticmethod
        def backward(ctx, grad_output):
            """
            This is the adjoint of the derivative only
            if g was the identity.
            """
            q, = ctx.saved_tensors
            q_ = q.detach().numpy()
            gxout, gyout = np.zeros_like(q_), np.zeros_like(q_)
            image_gradient(q_, gxout, gyout)
            grad = torch.tensor([gxout, gyout])
            result = grad*grad_output
            return (None, result)
    return CompositionAction.apply
