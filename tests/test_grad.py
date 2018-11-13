import numpy as np
import torch
from diffeopt.action.function import get_composition_action
from diffeopt.action.density import get_density_action
from diffeopt.group import DiffeoGroup

import pytest


def test_function():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    comp_16 = get_composition_action(small_shape, compute_id=True)

    idall = group.get_raw_identity(requires_grad=True)
    I16 = torch.randn(*small_shape, dtype=torch.float64)
    torch.autograd.gradcheck(
        comp_16,
        inputs=(I16, idall.double())
    )

def test_density():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    vol_16 = get_density_action(small_shape, compute_id=True)
    idall = group.get_raw_identity(requires_grad=True)
    x = torch.randn(*small_shape, dtype=torch.float64)
    torch.autograd.gradcheck(
        vol_16,
        inputs=(x, idall.double()),
        atol=1e-10,
        )


def test_identity():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    vol_16 = get_density_action(small_shape)
    idall_ = group.get_raw_identity()
    idall__ = idall_.double()
    x = torch.from_numpy(np.random.randn(*small_shape))
    res = vol_16(x, idall__)
    assert pytest.approx(torch.max((x-res).abs())) == 0

def test_one_jacobian():
    """
    Jacobian of translation is one.
    """
    shape = [16]*2
    group = DiffeoGroup(shape)
    act = get_density_action(shape)
    idall = group.get_raw_identity()
    idall[0] += 3
    x = torch.ones(shape, dtype=torch.float64)
    y = act(x, idall)
    assert pytest.approx(x.numpy()) == y.numpy()
