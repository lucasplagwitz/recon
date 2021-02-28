"""
02. Reconstruction I
================
This tutorial demonstrates the reconstruction of a
measurement obtained in computer tomography.
As mathematical construct the radon transformation is used here.
The implementations of skimage (radon, iradon) are used.

IN PROGRESS
"""

###############################################################################
# We create a scenario with a
import numpy as np
import matplotlib.pyplot as plt

from recon.utils import psnr
from recon.operator.ct_radon import CtRt
from recon.interfaces import Recon, ReconBregman, Smoothing, SmoothBregman

from matplotlib import image

# load image as pixel array
gt = image.imread("../data/phantom.png")
gt = gt/np.max(gt)
gt = gt

ntheta = 180
theta = np.linspace(0, 180, ntheta, endpoint=False)
sigma = 0.01
R = CtRt(gt.shape, center=[gt.shape[0]//2, gt.shape[1]//2], theta=theta)

y = R*gt.ravel()
y_max = np.max(y)

n = np.random.normal(0, sigma*y_max, size=y.shape)
y = y + n

x_rec = np.reshape(R.inv*y.ravel(), gt.shape)

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(gt, vmin=0, vmax=1)
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(np.reshape(y, R.image_dim).T)
axs[1].set_title('Data')
axs[1].axis('tight')
axs[2].imshow(x_rec, vmin=0, vmax=1)
axs[2].set_title("FBP - PSNR: "+str(psnr(gt, x_rec)))
axs[2].axis('tight')
fig.tight_layout()
plt.show()


lam = 15
rec = Recon(operator=R, domain_shape=gt.shape, reg_mode='tv', alpha=1, lam=lam, extend_pdhgm=True)
x_tv = rec.solve(data=y.ravel(), max_iter=1000, tol=1e-4)
plt.imshow(x_tv, vmin=0, vmax=1)

tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', lam=10, tau='calc')
fbp_smooth = tv_smoothing.solve(data=x_rec, max_iter=1000, tol=1e-4)


fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(gt, vmin=0, vmax=1)
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(x_tv, vmin=0, vmax=1)
axs[1].set_title("TV-Recon - PSNR: "+str(psnr(gt, x_tv)))
axs[1].axis('tight')
axs[2].imshow(fbp_smooth, vmin=0, vmax=1)
axs[2].set_title("FBP-Smooth - PSNR: "+str(psnr(gt, fbp_smooth)))
axs[2].axis('tight')
fig.tight_layout()
plt.show()
