import torch

from diffeopt.distance.information import information_distance

def test_zero_dist():
    """
    Information distance from image to itself is zero.
    """
    I = 1 + torch.rand(8,8) # make sure it is positive
    assert information_distance(I,I) == 0

def test_information_derivative():
    I0 = 1 + torch.rand(8,8)
    I1 = 1 + torch.rand(8,8)
    I1.requires_grad = True
    dist = information_distance(I0,I1)
    dist.backward()
    assert not torch.sum(torch.isnan(I1.grad))

