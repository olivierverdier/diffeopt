import numpy as np
import torch
from diffeopt.action.function import get_composition_action
from diffeopt.action.density import get_density_action
from diffeopt.utils import get_identity

import pytest


def test_function():
    small_shape = [16]*2
    comp_16 = get_composition_action(small_shape, compute_id=True)

    idall = get_identity(small_shape, requires_grad=True)
    I16 = torch.randn(*small_shape, dtype=torch.float64)
    torch.autograd.gradcheck(
        comp_16,
        inputs=(I16, idall.double())
    )

def test_density():
    small_shape = [16]*2
    vol_16 = get_density_action(small_shape, compute_id=True)
    idall = get_identity(small_shape, requires_grad=True)
    x = torch.randn(*small_shape, dtype=torch.float64)
    torch.autograd.gradcheck(
        vol_16,
        inputs=(x, idall.double()),
        atol=1e-10,
        )


def test_identity():
    small_shape = [16]*2
    vol_16 = get_density_action(small_shape)
    idall = get_identity(small_shape)
    idall_ = torch.tensor(idall, requires_grad=False)
    idall__ = idall_.double()
    x = torch.from_numpy(np.random.randn(*small_shape))
    res = vol_16(x, idall__)
    assert pytest.approx(torch.max((x-res).abs())) == 0
