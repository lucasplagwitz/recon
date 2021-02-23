"""
x01. Smoothing
================
This example ...
"""

###############################################################################
# We import ....

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from experimental.interfaces.smooth_bregman_satv import SmoothBregmanSATV
from recon.interfaces import Smoothing, SmoothBregman, SATV
from recon.solver.pd_hgm_tgv import PdHgmTGV
from recon.utils.utils import psnr

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

image = shepp_logan_phantom()
image[160:180, 140:160] = 1
for i in range(1, 20, 2):
    for j in range(1, 30, 4):
        image[190+j*2:192+j*2, 140+i*2:142+i*2] = np.random.uniform(1,1)

image[200:280, 240:260] = 1
for i in range(1, 20, 2):
    for j in range(1, 30, 3):
        image[290+j*2:292+j*2, 240+i*2:242+i*2] = np.random.uniform(1,1)

from scipy import misc

image = misc.ascent()

#image = rescale(image, scale=0.8, mode='reflect', multichannel=False)
image = image / np.max(image)
"""
image = np.zeros((64, 64))
image[10:50, 10:50] = 0.5
for i in range(1,20,4):
    for j in range(1, 20, 4):
        image[20+i:22+i, 20+j:22+j] = 1
"""
img = image / np.max(image)

gt = img

vmin, vmax = 0, np.max(img)

# create noisy image
sigma = 0.2 * vmax
n = np.random.normal(0, sigma, gt.shape)
noise_img = gt + n

f = plt.figure(figsize=(6, 3))
plt.gray()
f.add_subplot(1,2, 1)
plt.title("GT")
plt.axis('off')
plt.imshow(gt, vmin=vmin, vmax=vmax)
f.add_subplot(1, 2, 2)
plt.gray()
plt.title("Noisy")
plt.imshow(noise_img, vmin=vmin, vmax=vmax)
plt.axis('off')
plt.show(block=False)

###############################################################################
# TV-Regularization (with Bregman)
#
# TV smoothing small alpha

tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', alpha=0.3, lam=1, tau='calc')
u_tv = tv_smoothing.solve(data=noise_img, max_iter=5000, tol=1e-4)

breg_smoothing = SmoothBregman(domain_shape=gt.shape,
                               reg_mode='tv',
                               alpha=0.5,
                               lam=0.6,
                               tau='calc',
                               plot_iteration=False,
                               assessment=sigma * np.sqrt(np.prod(gt.shape)))

u_breg = breg_smoothing.solve(data=noise_img, max_iter=5000, tol=1e-4)

f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.gray()
plt.imshow(u_tv, vmin=vmin, vmax=vmax)
plt.title("TV - PSNR: "+str(psnr(gt, u_tv)))
f.add_subplot(1, 2, 2)
plt.imshow(u_breg, vmin=vmin, vmax=vmax)
plt.title("TV-Breg - PSNR: "+str(psnr(gt, u_breg)))
plt.axis('off')
plt.gray()
plt.show(block=False)

###############################################################################
# TV-Regularization via Bregman SATV
#
#
#

# TV smoothing small alpha
tv_smoothing = SmoothBregmanSATV(domain_shape=gt.shape,
                                 reg_mode='tv',
                                 lam=0.5,
                                 alpha=0.6,
                                 tau='calc',
                                 noise_sigma=sigma,
                                 plot_iteration=True,
                                 assessment=sigma*np.sqrt(np.prod(img.shape)))
u_tv = tv_smoothing.solve(data=noise_img, max_iter=5000, tol=1e-4)


f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.gray()
plt.imshow(noise_img, vmin=vmin, vmax=vmax)
plt.title("Noisy")
f.add_subplot(1, 2, 2)
plt.imshow(u_tv, vmin=vmin, vmax=vmax)
plt.title("BregSATV - PSNR"+ str(psnr(gt, u_tv)))
plt.axis('off')
plt.gray()
plt.show(block=False)


print("MIN-value: "+str(np.min(u_tv)))
print("MAX-value: "+str(np.max(u_tv)))



satv_obj = SATV(domain_shape=image.shape,
                reg_mode='tv',
                lam=0.5,
                alpha=0.6,
                plot_iteration=False,
                noise_sigma=sigma,
                assessment=sigma*np.sqrt(np.prod(image.shape)))
satv_solution = satv_obj.solve(noise_img, max_iter=5000, tol=1e-4)

f = plt.figure(figsize=(9, 3))
f.add_subplot(1, 3, 1)
plt.gray()
plt.axis('off')
plt.imshow(noise_img, vmin=0, vmax=vmax)
plt.title("Noisy - PSNR: "+str(psnr(image, noise_img)))
f.add_subplot(1, 3, 2)
plt.gray()
plt.imshow(satv_solution, vmin=0, vmax=vmax)
plt.title("SATV - PSNR: "+str(psnr(image, satv_solution)))
plt.axis('off')
f.add_subplot(1, 3, 3)
plt.gray()
plt.imshow(u_tv, vmin=vmin, vmax=vmax)
plt.title("BregSATV - PSNR"+ str(psnr(gt, u_tv)))
plt.axis('off')
plt.show()



"""
###############################################################################
# Since A also represents a convex functional, it can also be extended by Bregman.
plot_iteration = True
lam = 0.1
alpha = (0.1, 0.2)
assessment = 0.2 * np.max(img) * np.sqrt(np.prod(noise_img.shape))
pk = np.zeros(image.shape)
pk = pk.ravel()
i = 0

u = np.zeros(image.shape)
while True:
    print("current norm error: " + str(np.linalg.norm(u.ravel() - noise_img.ravel(), 2)))
    print("runs till norm <: " + str(assessment))

    solver = PdHgmTGV(alpha=alpha, lam=lam, mode='tgv', pk=pk)

    u_new = np.reshape(solver.solve(noise_img), gt.shape)

    if np.linalg.norm(u_new.ravel() - noise_img.ravel(), 2) < assessment:
        break

    u = u_new
    pk = pk - lam * (u.ravel() - noise_img.ravel())
    i = i + 1

    if plot_iteration:
        plt.gray()
        plt.imshow(u)
        plt.axis('off')
        plt.savefig('Bregman_TGV_iter' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
"""