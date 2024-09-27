from diffeopt.group.ddmatch.group import DiffeoGroup
import torch
from diffeopt.utils import get_random_diffeo
import numpy as np

def test_element():
    g = DiffeoGroup((16,16))
    g.identity()

def test_compose():
    g = DiffeoGroup((16,16))
    id1 = g.identity()
    id2 = g.identity()
    id3 = id1.compose(id2)
    assert torch.allclose(id1.forward, id3.forward)
    # TODO: add other tests here

def test_exponential():
    g = DiffeoGroup((16,16))
    z = torch.zeros((2,16,16), dtype=torch.float64)
    g.exponential(z)

def test_integ():
    g = DiffeoGroup((16,16))
    dim = 2
    vel = torch.randn(dim, dtype=torch.float64)
    velocity = vel.reshape(-1,1,1) * torch.ones((dim,16,16), dtype=torch.float64)
    g.exponential(velocity)
    # TODO: test that it is a translation

def test_inverse():
    group = DiffeoGroup((16,16))
    gen = torch.Generator()
    defm = get_random_diffeo(group, scale=.1, generator=gen)
    aid = defm.compose(defm.inverse())
    rid = group.get_raw_identity()
    assert np.allclose(rid.numpy(), aid.forward.numpy(), rtol=1e-9, atol=1e-1)



