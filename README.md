[![Build Status](https://github.com/olivierverdier/diffeopt/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/olivierverdier/diffeopt/actions/workflows/python_package.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/olivierverdier/diffeopt/graph/badge.svg?token=ZG233TWQQ8)](https://codecov.io/gh/olivierverdier/diffeopt)
![Python version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg?logo=python&logoColor=gold)

# Diffeopt: optimisation on diffeomorphisms

Optimisation on diffeomorphisms using Pytorch to compute the gradient automatically.

The general idea is to be able to minimise expressions of the form $g ↦ F(g · x_0)$, where 
- $g$ is a group element, typically a diffeomorphism
- $x_0$ is a *template*, typically either a density or a function (i.e., an image)
- $F$ is a *cost function*
- $g · x$ is an action (or representation) of the diffeomorphism group on densities or functions

This can be used to do direct matching or indirect matching, both with several kinds of regularisation.

Check out [this example notebook](https://gist.github.com/olivierverdier/2d30e409111376ff89b8e54ab82a8f9c) which illustrates the two kinds of matching.

<a href="https://gist.github.com/olivierverdier/2d30e409111376ff89b8e54ab82a8f9c"><img alt="deformation" src="https://raw.githubusercontent.com/olivierverdier/diffeopt/master/img/deformation.png" /></a>


## Direct Matching with Orbit Minimisation

Suppose that we have a *template* `I0` that we will use for the matching.
It should be a pytorch tensor.

We need a notion of a group:
```python
from diffeopt.group.ddmatch.group import DiffeoGroup
group = DiffeoGroup(I0.shape)
```

First, prepare the “network”, with one layer, which keeps a group element as a parameter, and computes one or several action on images.
Here, we want to compute, for the same group element, an action on function and one on densities:
```python
from diffeopt.group.ddmatch.representation import DensityRepresentation, FunctionRepresentation
from diffeopt.sum_representation import OrbitProblem
srep = OrbitProblem(FunctionRepresentation(group), DensityRepresentation(group))
```

Now we prepare an optimizer. It needs a learning rate and a cometric, as well as the network's parameters to be initialized:
```python
from diffeopt.cometric.laplace import get_laplace_cometric
from diffeopt.optim import GroupOptimizer
go = GroupOptimizer(srep.parameters(), lr=1e-1, cometric=get_laplace_cometric(group, s=2))
```

We now prepare the necessary variables to compute the loss function.

```python
from torch.nn import MSELoss
mse = MSELoss()
from diffeopt.utils import get_volume
vol = torch.ones(group.shape, dtype=torch.float64)/get_volume(group.shape)
```

The optimising loop is then as follows.
Note that the loss function can be anything you like.
Here, for direct matching, it depends on a target image `I1`.
```python
for i in range(2**9):
    go.zero_grad()
    # forward pass
    I_, vol_ = srep(I0, vol)
    # the following loss function can be anything you like
    loss = mse(I_, I1) + mse(vol_, vol)
    if not i % 2**6:
        print(i, loss)
    # compute momenta
    loss.backward()
    # update the group element
    go.step()
```
