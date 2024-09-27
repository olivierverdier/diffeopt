import numpy as np
import torch

def get_laplace_cometric(group, s=1):
    shape = group.shape
    idx, idy = group.get_raw_identity()
    lap = 4. - 2.*(np.cos(2.*np.pi*idx/shape[0]) + np.cos(2.*np.pi*idy/shape[1]))
    lap[0,0] = 1.
    lapinv = (1./lap)**s
    lap[0,0] = 0.
    lapinv[0,0] = 1.
    def cometric(momentum):
        fx = np.fft.fftn(momentum[0])
        fy = np.fft.fftn(momentum[1])
        fx *= lapinv
        fy *= lapinv
        vx = np.real(np.fft.ifftn(fx))
        vy = np.real(np.fft.ifftn(fy))
        return torch.from_numpy(np.array([vx,vy]))
    return cometric
