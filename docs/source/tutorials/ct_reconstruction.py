"""
02. Reconstruction I
================
This tutorial demonstrates the reconstruction of a
measurement obtained in computer tomography.
As mathematical construct the radon transformation is used here.
The implementations of skimage (radon, iradon) are used.
"""

###############################################################################
# We create a scenario with a
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from recon.utils import psnr
from recon.operator.ct_radon import CtRt
from recon.interfaces import Recon, ReconBregman, Smoothing, SmoothBregman

image = shepp_logan_phantom()
image = rescale(image, scale=0.2, mode='reflect', multichannel=False)
image = image / np.max(image)

ny, nx = image.shape

ntheta = 180
theta = np.linspace(0., 180, ntheta, endpoint=False)

sigma = 0.03

R = CtRt(image.shape, center=[image.shape[0]//2, image.shape[1]//2], theta=theta)
y = R*image.ravel()
n = np.random.normal(0, sigma*np.max(y), size=y.shape)
y = y + n

x_rec = np.reshape(R.inv*y.ravel(), image.shape)

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

###############################################################################
# Similar to the Smoothing Tutorial a saddle point problem between
# data fidelity and regularization is solved. For this purpose the Radon
# operator is simply passed to the data term. The Recon interface takes care
# of everything for the user.
# We add a quick comparison to the solution that arises
# when one should first reconstruct and then regularize.

rec = Recon(operator=R, domain_shape=(ny, nx), reg_mode='tv', alpha=0.05, lam=1, tau='calc')
x_tv = rec.solve(data=y.ravel(), max_iter=1000, tol=1e-5)

smooth = Smoothing(domain_shape=image.shape, reg_mode='tv', alpha=0.05, lam=1, tau='calc')
x_succession = smooth.solve(R.inv*y.ravel(), max_iter=1000, tol=1e-4)


fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(np.reshape(x_tv, R.domain_dim), vmin=0, vmax=1, cmap='gray')
axs[1].set_title('TV-Recon - PSNR:'+str(psnr(image, x_tv)))
axs[1].axis('tight')
axs[2].imshow(np.reshape(x_succession, R.domain_dim), vmin=0, vmax=1, cmap='gray')
axs[2].set_title("Smooth $R^{-1}$*y - PSNR: "+str(psnr(image, x_succession)))
axs[2].axis('tight')
fig.tight_layout()

plt.show()

###############################################################################
# Bregman versions

rec = ReconBregman(operator=R,
                   domain_shape=image.shape,
                   reg_mode='tv',
                   alpha=0.6,
                   lam=1,
                   assessment=sigma*np.max(y)*np.sqrt(np.prod(n.shape)),
                   tau='calc')
breg_tv = rec.solve(data=y.ravel(), max_iter=1000, tol=1e-4)

breg_smoothing = SmoothBregman(domain_shape=image.shape,
                               reg_mode='tv',
                               alpha=0.6,
                               lam=1,
                               tau='calc',
                               plot_iteration=False,
                               assessment=np.linalg.norm(R.inv*n.ravel(), 2))

u_breg = breg_smoothing.solve(data=R.inv*y.ravel(), max_iter=1000, tol=1e-4)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')
axs[0].set_title('Model')
axs[0].axis('tight')
axs[1].imshow(np.reshape(breg_tv, R.domain_dim), vmin=0, vmax=1, cmap='gray')
axs[1].set_title('TV-Breg - PSNR:'+str(psnr(image, breg_tv)))
axs[1].axis('tight')
axs[2].imshow(np.reshape(u_breg, R.domain_dim), vmin=0, vmax=1, cmap='gray')
axs[2].set_title("Breg $R^{-1}$*y - PSNR: "+str(psnr(image, u_breg)))
axs[2].axis('tight')
fig.tight_layout()

plt.show()

###############################################################################
# Conclusion
# Further tests will follow. It seems that the L2 standard is not the best
# choice for the radon-sinogram space.
# ... -> test L1-Norm