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
from recon.interfaces import Recon
from experimental.interfaces.recon_satv import ReconSATV
from recon.utils.images import objects_differ_scales


from skimage.data import shepp_logan_phantom
from skimage.transform import rescale


image = objects_differ_scales()
image = np.zeros((128,128))
image[10:110, 10:110] = 1
for i in range(1,90,4):
    for j in range(1, 20, 4):
        image[15+i:17+i, 15+j:17+j] = 0

image[20:45, 50:100] = 0.0
image[65:85, 50:100] = 0.25
image[95:105, 50:100] = 0.5

#image = rescale(image, scale=0.7, mode='reflect', multichannel=False)
from skimage import io
image = io.imread("myImage.png")
image = image/np.max(image)

"""
image = shepp_logan_phantom()
image[160:180, 140:160] = 1
for i in range(1, 20, 2):
    for j in range(1, 30):
        image[190+j*2:192+j*2, 140+i*2:142+i*2] = np.random.uniform(0.5,1)
image[200:280, 240:260] = 1
for i in range(1, 20, 2):
    for j in range(1, 30):
        image[290+j*2:292+j*2, 240+i*2:242+i*2] = np.random.uniform(1,1)
image = rescale(image, scale=0.45, mode='reflect', multichannel=False)
image = image / np.max(image)
"""

image = image

"""
image = np.zeros((64, 64))
image[10:50, 10:50] = 0.5
for i in range(1,20,4):
    for j in range(1, 20, 4):
        image[20+i:22+i, 20+j:22+j] = 1
"""
image = image / np.max(image)

ny, nx = image.shape

ntheta = 150
theta = np.linspace(0., 180, ntheta, endpoint=False)

sigma = 3

R = CtRt(image.shape, center=[image.shape[0]//2, image.shape[1]//2], theta=theta)
y = R*image.ravel()
n = np.random.normal(0, sigma, size=y.shape)
y = y + n

x_rec = np.reshape(R.inv*y.ravel(), image.shape)

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(np.reshape(y, R.image_dim), cmap='gray')
axs[1].set_title('Data')
axs[1].axis('tight')
axs[2].imshow(x_rec, cmap='gray', vmin=0, vmax=1)
axs[2].set_title("Reconstruction - PSNR: "+str(psnr(image, x_rec)))
axs[2].axis('tight')
fig.tight_layout()

plt.show()


#############################
#
#
"""
rec = Recon(operator=R, domain_shape=(ny, nx), reg_mode='tv', alpha=0.2, lam=1)
x_tv = rec.solve(data=y.ravel(), max_iter=400, tol=1e-4)

#rec = Recon(operator=R, domain_shape=(ny, nx), reg_mode='tgv', alpha=(0.1, 0.2), lam=1)
#x_tgv = rec.solve(data=y.ravel(), max_iter=400, tol=1e-4)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(x_tv, vmin=0, vmax=1, cmap='gray')
axs[0].set_title("TV - PSNR: "+str(psnr(image, x_tv)))
axs[0].axis('tight')
axs[1].imshow(x_rec, cmap='gray', vmin=0, vmax=1)
axs[1].set_title("Reconstruction - PSNR: "+str(psnr(image, x_rec)))
axs[1].axis('tight')
fig.tight_layout()
"""

plt.show()
#############################
#
#

lam = 1.2*np.ones(image.shape)
rc_satv = ReconSATV(operator=R,
                    domain_shape=(ny, nx),
                    reg_mode='tv',
                    lam=lam,
                    alpha=1,
                    tau='calc',
                    noise_sigma=sigma,
                    plot_iteration=True,
                    assessment= sigma*np.sqrt(np.prod(y.shape)))
rc_satv.w = R.inv*y.ravel()

rec = Recon(operator=R,
            domain_shape=(ny, nx),
            reg_mode='tv',
            lam=0.1)
rec.extend_pdhgm = True
rc_satv.first_over = rec.solve(data=y.ravel(), max_iter=1500, tol=1e-4)

plt.imshow(rc_satv.first_over)
plt.show()

rc_satv = rc_satv.solve(data=y.ravel(), max_iter=1000, tol=1e-4)

plt.imshow(np.reshape(rc_satv, R.domain_dim), vmin=0, vmax=1, cmap='gray')
plt.title("SATV - PSNR: "+str(psnr(image, rc_satv)))
plt.show()

print("MIN: "+str(np.min(rc_satv)))
print("MAX: "+str(np.max(rc_satv)))