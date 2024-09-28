import torch

def information_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    The information distance is the angle between the function $F = \sqrt{f}$ and $G = \sqrt{g}$.
    In other words, it is obtained by
    \\[
    d(f,g) := \arccos\Big( \frac{(F,G)}{\|F\|\|G\|} \Big)
    \\]
    where $\|F\| \|G\| = \sqrt{\int f \int g}$, $(F,G) = \int \sqrt{fg}$.
    """
    volume = torch.sqrt(torch.sum(x)*torch.sum(y))
    product = torch.sum(torch.sqrt(x*y))/volume
    return torch.acos(product)



