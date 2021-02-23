import numpy as np

import pylops
from pylops import Gradient
from recon.solver import PdHgm
from recon.terms import DatanormL2, IndicatorL2
from pylops import Smoothing2D
import matplotlib.pyplot as plt
from recon.interfaces import Recon, Smoothing
from recon.utils import psnr
from recon.utils.utils import power_method

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

image = shepp_logan_phantom()
image = rescale(image, scale=0.3, mode='reflect', multichannel=False)
image = image / np.max(image)

#image = np.zeros((64, 64))
#image[8:24, 8:24] = 0.8
#image[16:20, 16:20] = 1
#image[38:54, 38:54] = 0.8
#image[46:50, 46:50] = 1

x = image
x = x / x.max()
ny, nx = x.shape

ID = pylops.Identity(np.prod(x.shape))
from recon.operator import CtRt

def radoncurve(x, r, theta):
    return (r - ny//2)/(np.sin(np.deg2rad(theta))+1e-15) + np.tan(np.deg2rad(90 - theta))*x  + ny//2

ntheta = 20
theta = np.linspace(0., 180., ntheta, endpoint=False)

RLop = \
    pylops.signalprocessing.Radon2D(np.arange(ny), np.arange(nx),
                                    theta, kind=radoncurve,
                                    centeredh=True, interp=False,
                                    engine='numba', dtype='float64')

#norm = np.abs(np.asscalar((RLop.H * RLop).eigs(neigs=1, symmetric=True, largest=True, uselobpcg=True)))
#print(norm)


R = CtRt(image.shape, center=[image.shape[0]//2, image.shape[1]//2], theta=theta)

norm = power_method(R, R.H, max_iter=100)
print("----"+str(norm))

#R = CtRt(image.shape, center=[image.shape[0]//2, image.shape[1]//2], theta=theta)#, norm=np.sqrt(norm))
y = R*image.ravel()
sigma = 0.01
maxmax = np.max(y)
n = np.random.normal(0, sigma*np.max(y), size=y.shape)
y = y + n

plt.imshow(np.reshape(y, R.image_dim))
plt.show()

plt.imshow(np.reshape(R.inv*y, image.shape))
plt.title("FBP - PSNR: "+str(psnr(x.ravel(), R.inv*y)))
plt.show()

sino = y
itera = 1
er = np.zeros(x.shape)
while itera<20:
    #lam = 0.00001*np.ones(sino.shape)*itera-itera*0.000001*np.reshape(R*er.ravel(), sino.shape)
    """
    rec = Recon(operator=R,
                domain_shape=x.shape,
                reg_mode='tgv',
                tau=1/np.sqrt(norm)*0.9,
                alpha=(1, 2), lam=0.1, extend_pdhgm=True)

    n = np.random.normal(0, 0.1, size=x.shape)
    noise_img = x+n
    x_tv = rec.solve(data=y.ravel(), max_iter=250, tol=1e-4) #-R*er.ravel()
    direct = x_tv.ravel()
    #direct = RLop*sino.T
    """
    tv_smoothing = Smoothing(domain_shape=image.shape, reg_mode='tv', alpha=0.1, lam=0.3, tau='calc')
    x_tv = tv_smoothing.solve(data=R.inv*y, max_iter=2000, tol=1e-4)

    plt.imshow(np.reshape(x_tv, image.shape), vmax=np.max(x))
    plt.title("TGV - PSNR: "+str(psnr(x, x_tv)))
    plt.show()

    while itera<20:
        er = np.abs((R.inv*(R*(x_tv.ravel())))-x_tv.ravel())
        Sop = Smoothing2D(nsmooth=[5, 5], dims=image.shape, dtype='float64')
        er = Sop*er.ravel()
        er = er / np.max(er)
        er = 20*(er+0.5)**6
        #er[er<0.5] = 0.5
        plt.imshow(np.reshape(er.ravel(), image.shape), vmax=np.max(x))
        plt.show()

        itera += 1
        ne1 = np.linalg.norm(np.abs(y-R*x_tv.ravel()), 2)
        ne2 = sigma*maxmax*np.sqrt(np.prod(y.shape))
        print("IS: "+str(ne1)+"- run till -"+str(ne2))
        if ne1 < ne2:
            break

        rec = Recon(operator=R,
                    domain_shape=x.shape,
                    reg_mode='tik',
                    tau=1 / np.sqrt(norm) * 0.9,
                    data=x_tv.ravel(),
                    alpha=(er, 0.2), lam=0.1, extend_pdhgm=True)

        x_tik = rec.solve(data=y.ravel(), max_iter=250, tol=1e-4)

        plt.imshow(np.reshape(x_tik, image.shape), vmax=np.max(x))
        plt.title("TIK - PSNR: " + str(psnr(x, x_tik)))
        plt.show()

        x_tv = x_tik




"""
def radoncurve(x, r, theta):
    return (r - ny//2)/(np.sin(np.deg2rad(theta))+1e-15) + np.tan(np.deg2rad(90 - theta))*x  + ny//2

ntheta = 150
theta = np.linspace(0., 180., ntheta, endpoint=False)

RLop = \
    pylops.signalprocessing.Radon2D(np.arange(ny), np.arange(nx),
                                    #theta, kind=radoncurve,
                                    centeredh=True, interp=False,
                                    engine='numba', dtype='float64')

sino = RLop.H * x.T.ravel()
sino = sino.reshape(ntheta, ny).T.ravel()

ID = pylops.Identity(np.prod(x.shape))


rec = Recon(operator=RLop.H, domain_shape=x.shape, reg_mode='tv', alpha=(2.5, 0.1), lam=1)

n = np.random.normal(0, 0.1, size=x.shape)
noise_img = x+n
#x_tv = rec.solve(data=sino.ravel(), max_iter=25, tol=1e-4)
#direct = x_tv.ravel()
direct = RLop*sino.T

plt.imshow(sino.reshape(ntheta, ny).T)
plt.show()

plt.imshow(np.reshape(x.T, image.shape))
plt.title("TGV - PSNR: "+str(psnr(x.ravel(), direct)))
plt.show()
"""
"""
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
"""