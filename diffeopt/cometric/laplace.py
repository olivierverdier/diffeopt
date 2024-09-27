import numpy as np
import torch

def get_fourier_cometric(group, s):
    shape = group.shape
    idx, idy = group.get_raw_identity()
    lap = 4. - 2.*(torch.cos(2.*torch.pi*idx/shape[0]) + torch.cos(2.*torch.pi*idy/shape[1]))
    lap[0,0] = 1.
    lapinv = (1./lap)**s
    lap[0,0] = 0.
    lapinv[0,0] = 1.
    return lapinv

def get_laplace_cometric(group, s=1):
    def cometric(momentum):
    lapinv = get_fourier_cometric(group, s)
        fx = torch.fft.fftn(momentum[0])
        fy = torch.fft.fftn(momentum[1])
        fx *= lapinv
        fy *= lapinv
        vx = torch.real(torch.fft.ifftn(fx))
        vy = torch.real(torch.fft.ifftn(fy))
        return torch.stack([vx,vy])
    return cometric
