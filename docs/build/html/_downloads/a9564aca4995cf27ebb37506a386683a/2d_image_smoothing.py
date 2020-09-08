"""
01. Smoothing
================
This example ...
"""

###############################################################################
# We import ....

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from recon.interfaces import Smoothing

img = misc.ascent()
img = img/np.max(img)
gt = img

vmin, vmax = 0, 1

# create noisy image
sigma = 0.2
n = sigma*np.random.uniform(-1, 1, gt.shape)
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
# TV-Regularization and Tikhonov
#
#
#

# TV smoothing small alpha
tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', alpha=0.3)
u_tv = tv_smoothing.solve(data=noise_img, max_iter=450, tol=10**(-5))

# Tikhonov smoothing -> with lam = 1 => alpha > 1 we decrease lam instead.
tikh_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tikhonov', lam=0.1, alpha=1, tau=0.1)
u_tik = tikh_smoothing.solve(data=noise_img, max_iter=450, tol=10**(-5))

f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.gray()
plt.imshow(u_tik, vmin=vmin, vmax=vmax)
plt.title("Tikhonov")
f.add_subplot(1, 2, 2)
plt.imshow(u_tv, vmin=vmin, vmax=vmax)
plt.title("TV")
plt.axis('off')
plt.gray()
plt.show(block=False)

###############################################################################
# 1D compare with [gt, noise, tikhonov, tv]

x_min = 84
x_max = 155
y = 20
plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
plt.plot(range(x_min, x_max), u_tik[x_min:x_max,y], color="blue", label="Tikhonov")
plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
plt.plot(range(x_min, x_max), u_tv[x_min:x_max,y], color="green", label="TV")
plt.legend(loc="lower left")
plt.plot(bbox_inches='tight', pad_inches=0)
plt.show()

###############################################################################
# Later on....
# Bregman is not yet adjusted.
"""
# bregman iteration
breg_smoothing = SmoothBregman(domain_shape=gt.shape,
                               reg_mode='tv',
                               alpha=1.1,
                               tau=0.0782,
                               plot_iteration=True,
                               assessment=0.6 * sigma*np.max(abs(gt.ravel())) * np.sqrt(np.prod(gt.shape)) )
u_breg = breg_smoothing.solve(data=noise_img, max_iter=150, tol=5*10**(-6))
draw_images(u_breg, '2d_smoothing_bregman.png', vmin=0, vmax=np.max(gt))


# 1d comparisson with [gt, noise, bregman_tv, tv, tikhonov]
x_min = 84
x_max = 155
y = 20
plt.plot(range(x_min, x_max), u_tik[x_min:x_max,y], color="darkcyan", label="Tikhonov")
plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
plt.plot(range(x_min, x_max), u_tv[x_min:x_max,y], color="green", label="TV")
plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
plt.plot(range(x_min, x_max), u_breg[x_min:x_max,y], color="blue", label="BregTV")
plt.legend(loc="lower left")
plt.savefig(data_output_path+'2d_smoothing_1d_comp_2.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()
"""