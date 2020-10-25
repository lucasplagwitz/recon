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
from recon.interfaces import Smoothing, SmoothBregman
from recon.solver.pd_hgm_extend import PdHgmTGV
from recon.utils.utils import psnr


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
tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', alpha=0.1, lam=0.3, tau='calc')
u_tv = tv_smoothing.solve(data=noise_img, max_iter=5000, tol=1e-4)

breg_smoothing = SmoothBregman(domain_shape=gt.shape,
                               reg_mode='tv',
                               alpha=0.5,
                               lam=0.1,
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
                                 lam=0.1,
                                 alpha=0.5,
                                 tau='calc',
                                 noise_sigma=sigma,
                                 assessment=sigma*np.sqrt(np.prod(img.shape)),
                                 plot_iteration=True)
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