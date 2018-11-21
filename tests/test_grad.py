import numpy as np
import torch
from diffeopt.action.function import get_composition_action
from diffeopt.action.density import get_density_action
from diffeopt.group import DiffeoGroup

mse = torch.nn.MSELoss()

import pytest


def test_grad_img():
    small_shape = [16]*2
    comp_16 = get_composition_action(small_shape)
    group = DiffeoGroup(small_shape)
    v = group.zero()
    v[0] = 1.5
    v[1] = .5
    elt = group.exponential(v)
    I16 = torch.ones(*small_shape, dtype=torch.float64)
    I16.requires_grad = True
    torch.autograd.gradcheck(comp_16, inputs=(I16, elt))

def test_grad_loss():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    comp_16 = get_composition_action(small_shape)

    idall = group.get_raw_identity(requires_grad=True)
    I16 = torch.randn(*small_shape, dtype=torch.float64)
    loss = mse(comp_16(I16, idall), torch.zeros(small_shape, dtype=torch.float64))
    torch.autograd.backward(loss)

def test_function():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    comp_16 = get_composition_action(small_shape, compute_id=True)

    idall = group.get_raw_identity(requires_grad=True)
    I16 = torch.randn(*small_shape, dtype=torch.float64)
    torch.autograd.gradcheck(
        comp_16,
        inputs=(I16, idall)
    )

def test_density():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    vol_16 = get_density_action(small_shape, compute_id=True)
    idall = group.get_raw_identity(requires_grad=True)
    x = torch.randn(*small_shape, dtype=torch.float64)
    torch.autograd.gradcheck(
        vol_16,
        inputs=(x, idall),
        atol=1e-10,
        )


def test_identity():
    small_shape = [16]*2
    group = DiffeoGroup(small_shape)
    vol_16 = get_density_action(small_shape)
    idall_ = group.element()
    x = torch.from_numpy(np.random.randn(*small_shape))
    res = vol_16(x, idall_)
    assert pytest.approx(torch.max((x-res).abs())) == 0

def test_one_jacobian():
    """
    Jacobian of translation is one.
    """
    shape = [16]*2
    group = DiffeoGroup(shape)
    act = get_density_action(shape)
    idall = group.element()
    vel = group.zero()
    vel[0] += 3
    trans = group.exponential(vel)
    x = torch.ones(shape, dtype=torch.float64)
    y = act(x, trans)
    assert pytest.approx(x.numpy()) == y.numpy()

