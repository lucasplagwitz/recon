"""
05. Spatially Adapted Total Variation
================

SATV

"""
import numpy as np
import matplotlib.pyplot as plt
from recon.interfaces import SATV, Smoothing

size, small_size = 256, 200
image = np.reshape(np.array([(x/size) for x in range(size)]*size), (size, size))
image[28:small_size+28, 28:small_size+28] = \
    np.reshape(np.array([(1-x/small_size)for x in range(small_size)]*small_size), (small_size, small_size))

some_dots = [(100, 105), (110, 115), (120, 125), (130, 135), (140, 145)]

for dot0 in some_dots:
    for dot1 in some_dots:
        image[dot0[0]: dot0[1], dot1[0]: dot1[1]] = 1

image = image / np.max(image) * 255

noisy_image = image + np.random.normal(0, 0.2*np.max(image), size=(size, size))

# TV smoothing small alpha
tv_smoothing = Smoothing(domain_shape=image.shape, reg_mode='tv', alpha=1, lam=0.005)
u_tv = tv_smoothing.solve(data=noisy_image, max_iter=550, tol=10**(-5))


f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.gray()
plt.imshow(image, vmin=0, vmax=np.max(image))
plt.title("GT")
f.add_subplot(1, 2, 2)
plt.imshow(u_tv, vmin=0, vmax=np.max(image))
plt.title("TV")
plt.axis('off')
plt.gray()
plt.show(block=False)

###############################################################################
# ...

satv_obj = SATV(domain_shape=image.shape,
                     reg_mode='tv',
                     lam=0.0001,
                     tau=0.4,
                     noise_sigma=0.2*np.max(image),
                     assessment=0.2*np.max(image)*np.sqrt(np.prod(image.shape)))
satv_solution = satv_obj.solve(noisy_image, max_iter=550)

f = plt.figure(figsize=(9, 3))
f.add_subplot(1, 3, 1)
plt.gray()
plt.axis('off')
plt.imshow(noisy_image, vmin=0, vmax=np.max(image))
plt.title("Noisy")
f.add_subplot(1, 3, 2)
plt.gray()
plt.imshow(satv_solution, vmin=0, vmax=np.max(image))
plt.title("SATV Denoising")
plt.axis('off')
f.add_subplot(1, 3, 3)
plt.gray()
plt.imshow(np.reshape(satv_obj.lam, image.shape))
plt.title("SATV-weight $\lambda$")
plt.axis('off')
plt.show()