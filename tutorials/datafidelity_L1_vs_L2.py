"""
06. L1 vs. L2 Fidelity-Term
===========================
This tutorial shows the difference between
two fidelity norms with different underlying noise types.
Here using the example of Salt-And-Pepper and Gaussian noise.

"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from recon.utils.utils import psnr
from recon.interfaces import Smoothing

img = misc.face(gray=True)[256:768,256:768]
img = img/np.max(img)
gt = img

vmin, vmax = 0, 1
sigma = 0.2 * vmax

###############################################################################
# Create two noisy images.
#

def sp_noise(image):
    """Add salt and pepper noise to image."""
    s_vs_p = 0.5
    amount = 0.1
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out


noise_img_sp = sp_noise(gt)
noise_img_white = gt + np.random.normal(0, sigma, size=gt.shape)

###############################################################################
# Application of the various fidelity norms.
#

# L1-fidelity
tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', norm='L1', alpha=1, lam=1, tau='calc')
u_L1_sp = tv_smoothing.solve(data=noise_img_sp, max_iter=2000, tol=1e-4)

tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', norm='L1', alpha=0.8, lam=1, tau='calc')
u_L1_white = tv_smoothing.solve(data=noise_img_white, max_iter=2000, tol=1e-4)


# L2-fidelity
tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', norm='L2', alpha=0.1, lam=0.5, tau='calc')
u_L2_sp = tv_smoothing.solve(data=noise_img_sp, max_iter=2000, tol=1e-4)

tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', norm='L2', alpha=0.1, lam=0.6, tau='calc')
u_L2_white = tv_smoothing.solve(data=noise_img_white, max_iter=2000, tol=1e-4)

# plot
f = plt.figure(figsize=(9, 6))
# Salt-And-Pepper
f.add_subplot(2, 3, 1)
plt.axis('off')
plt.gray()
plt.imshow(noise_img_sp, vmin=vmin, vmax=vmax)
plt.title("S&P-Noise-PSNR: "+str(psnr(gt, noise_img_sp)))

f.add_subplot(2, 3, 2)
plt.imshow(u_L1_sp, vmin=vmin, vmax=vmax)
plt.title("L1 - PSNR: "+str(psnr(gt, u_L1_sp)))
plt.axis('off')
plt.gray()

f.add_subplot(2, 3, 3)
plt.imshow(u_L2_sp, vmin=vmin, vmax=vmax)
plt.title("L2 - PSNR: "+str(psnr(gt, u_L2_sp)))
plt.axis('off')
plt.gray()

# Gaussian
f.add_subplot(2, 3, 4)
plt.axis('off')
plt.gray()
plt.imshow(noise_img_white, vmin=vmin, vmax=vmax)
plt.title("White-Noise - PSNR: "+str(psnr(gt, noise_img_white)))

f.add_subplot(2, 3, 5)
plt.imshow(u_L1_white, vmin=vmin, vmax=vmax)
plt.title("L1 - PSNR: "+str(psnr(gt, u_L1_white)))
plt.axis('off')
plt.gray()

f.add_subplot(2, 3, 6)
plt.imshow(u_L2_white, vmin=vmin, vmax=vmax)
plt.title("L2 - PSNR: "+str(psnr(gt, u_L2_white)))
plt.axis('off')
plt.gray()

plt.show(block=False)
