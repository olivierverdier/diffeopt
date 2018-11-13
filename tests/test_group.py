from diffeopt.group import DiffeoGroup
import torch

def test_element():
    g = DiffeoGroup((16,16))
    g.element()

def test_compose():
    g = DiffeoGroup((16,16))
    id1 = g.element()
    id2 = g.element()
    id3 = id1.compose(id2)
    assert torch.allclose(id1.data, id3.data)
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



