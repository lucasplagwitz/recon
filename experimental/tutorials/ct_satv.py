"""
x02. Computer Tomographie with SATV
================
This example ...
Problem setting from PyLops: https://pylops.readthedocs.io/en/latest/tutorials/ctscan.html?highlight=ct
"""

###############################################################################
# We import ....

import numpy as np
import matplotlib.pyplot as plt
from recon.operator.ct_radon import CtRt

from recon.utils.utils import psnr
from experimental.interfaces.recon_satv import ReconSATV


size, small_size = 256, 200
image = np.reshape(np.array([(x/size) for x in range(size)]*size), (size, size))
image[28:small_size+28, 28:small_size+28] = \
    np.reshape(np.array([(1-x/small_size)for x in range(small_size)]*small_size), (small_size, small_size))

some_dots = [(i*10, i*10+5) for i in range(5, 15)]
some_dots += [(i*10, i*10+2) for i in range(15, 21)]
some_dots += [(i*10+3, i*10+5) for i in range(15, 21)]

for dot0 in some_dots:
    for dot1 in some_dots:
        image[dot0[0]: dot0[1], dot1[0]: dot1[1]] = 1

image = image[0:100, 150:250]

image = image / np.max(image)

ny, nx = image.shape

ntheta = 180
theta = np.linspace(0., 180, ntheta, endpoint=False)

sigma = 0.02

R = CtRt(image.shape, center=[image.shape[0]//2, image.shape[1]//2], theta=theta)
y = R*image.ravel()
n = np.random.normal(0, sigma*np.max(y), size=y.shape)
y = y + n

x_rec = np.reshape(R.H*y.ravel(), image.shape)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(np.reshape(y, R.image_dim), cmap='gray')
axs[1].set_title('Data')
axs[1].axis('tight')
axs[2].imshow(x_rec, cmap='gray')
axs[2].set_title("Reconstruction - PSNR: "+str(psnr(image, x_rec)))
axs[2].axis('tight')
fig.tight_layout()

plt.show()


#############################
#
#
"""
rec = Recon(operator=R, domain_shape=(ny, nx), reg_mode='tv', alpha=0.2, lam=0.4, tau='calc')
x_tv = rec.solve(data=y.ravel(), max_iter=5000, tol=1e-4)

rec = Recon(operator=R, domain_shape=(ny, nx), reg_mode='tikhonov', alpha=1.0, lam=0.4, tau='calc')
x_tik = rec.solve(data=y.ravel(), max_iter=5000, tol=1e-4)

rec_breg = ReconBregman(operator=R,
                        domain_shape=(ny, nx),
                        reg_mode='tv',
                        alpha=0.3, lam=0.1,
                        tau='calc',
                        plot_iteration=False,
                        assessment=sigma*np.max(y)*np.sqrt(np.prod(y.shape)))
x_breg = rec_breg.solve(data=y.ravel(), max_iter=5000, tol=1e-4)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(x_tv, vmin=0, vmax=1, cmap='gray')
axs[0].set_title("TV - PSNR: "+str(psnr(image, x_tv)))
axs[0].axis('tight')
axs[1].imshow(np.reshape(x_tik, R.domain_dim), vmin=0, vmax=1, cmap='gray')
axs[1].set_title("Tikhonov - PSNR: "+str(psnr(image, x_tik)))
axs[1].axis('tight')
axs[2].imshow(np.reshape(x_breg, R.domain_dim), vmin=0, vmax=1, cmap='gray')
axs[2].set_title("Bregman - PSNR: "+str(psnr(image, x_breg)))
axs[2].axis('tight')
fig.tight_layout()

plt.show()

"""
#############################
#
#
sigma_x = np.sum(np.abs(R.H*y.ravel() - image.ravel()))/np.prod(image.shape)

lam = 0.1*np.ones(image.shape)
rc_satv = ReconSATV(operator=R,
                    domain_shape=(ny, nx),
                    reg_mode='tv',
                    lam=lam,
                    alpha=1,
                    tau='calc',
                    noise_sigma=sigma*np.max(y),
                    plot_iteration=True,
                    assessment= sigma*np.max(y)*np.sqrt(np.prod(image.shape)))
rc_satv.w = R.H*y.ravel()
rc_satv = rc_satv.solve(data=y.ravel(), max_iter=5000, tol=1e-4)

plt.imshow(np.reshape(rc_satv, R.domain_dim), vmin=0, vmax=1, cmap='gray')
plt.title("SATV - PSNR: "+str(psnr(image, rc_satv)))
plt.show()