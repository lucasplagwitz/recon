import numpy as np

import pylops
from pylops import Gradient
from recon.solver import PdHgm
from recon.terms import DatanormL2, IndicatorL2

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

image = shepp_logan_phantom()
image = rescale(image, scale=0.2, mode='reflect', multichannel=False)
image = image / np.max(image)

x = image
x = x / x.max()
ny, nx = x.shape

def radoncurve(x, r, theta):
    return (r - ny//2)/(np.sin(np.deg2rad(theta))+1e-15) + np.tan(np.deg2rad(90 - theta))*x  + ny//2

ntheta = 180
theta = np.linspace(0., 180., ntheta, endpoint=False)

RLop = \
    pylops.signalprocessing.Radon2D(np.arange(ny), np.arange(nx),
                                    theta, kind=radoncurve,
                                    centeredh=True, interp=False,
                                    engine='numba', dtype='float64')

sino = RLop.H * x.T.ravel()
sino = sino.reshape(ntheta, ny).T

alpha = 0.1
K = Gradient(x.shape, edge=True, dtype='float64', kind='backward', sampling=1) * RLop.H.H
norm = np.abs(np.asscalar((K.H*K).eigs(neigs=1, symmetric=True, largest=True, uselobpcg=True)))
fac = 0.99
tau = fac * np.sqrt(1 / norm)
print(tau)
F_star = IndicatorL2(x.shape,
                     len(x.shape),
                     prox_param=tau,
                     upper_bound=alpha)
G = DatanormL2(image_size=sino.shape, prox_param=tau, lam=1, data=sino.ravel())


solver = PdHgm(K, F_star, G)
solver.max_iter=40
x = solver.solve()

import matplotlib.pyplot as plt

plt.imshow(np.reshape(RLop.H.H*solver.x, image.shape))
plt.show()