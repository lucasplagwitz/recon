"""
04. Total Generalized Variation
===============================
Now we take a step deeper into Total Variation based regularization.

We focus on concepts from different papers.
Mainly we use for numerical access:
    Knoll, Bredis, Pock: Second Order Total Generalized Variation (TGV) for MRI


"""

###############################################################################
# The first order Total Variation got some problems with smooth edges.
# See following noisy example with the TV-Regularization.
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces import Smoothing
size = 256
small_size =164
#very_small_size = 150

# build image
image = np.reshape(np.array([(x/size) for x in range(size)]*size), (size, size))
image[40:small_size+40, 40:small_size+40] = \
    np.reshape(np.array([(1-x/small_size)for x in range(small_size)]*small_size), (small_size, small_size))
image[120:125, 100:120] = 1
image[100:120, 120:125] = 1
#image[140:140+very_small_size, 140:140+very_small_size] = \
#np.reshape(np.array([(x/very_small_size)for x in range(very_small_size)]*very_small_size), (very_small_size, very_small_size))

noise_image = image + np.random.normal(0, 0.2, size=image.shape)

tv_denoising = Smoothing(domain_shape=image.shape, reg_mode='tv', lam=0.5, alpha=0.25)
tv_solution = tv_denoising.solve(noise_image, max_iter=500)

f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.gray()
plt.axis('off')
plt.imshow(noise_image, vmin=0, vmax=1)
plt.title("Noisy")
f.add_subplot(1, 2, 2)
plt.gray()
plt.imshow(tv_solution, vmin=0, vmax=1)
plt.title("TV based denoising")
plt.axis('off')
plt.show()



###############################################################################
# To avoid strong stair-casing effects, we introduce the Total Generalized Variation.
# At this point there is no interface for second order TV. We implement it direct with the
# adapted Primal-Dual algorithm.

from recon.solver.pd_hgm_extend import PdHgmTGV

# TGV smoothing small alpha
alpha = (0.25, 0.5)
solver = PdHgmTGV(alpha=alpha, lam=0.5)
tgv_solution = np.reshape(solver.solve(noise_image), (size, size))

f = plt.figure(figsize=(9, 3))
f.add_subplot(1, 3, 1)
plt.gray()
plt.axis('off')
plt.imshow(image, vmin=0, vmax=1)
plt.title("Original")
f.add_subplot(1, 3, 2)
plt.gray()
plt.axis('off')
plt.imshow(tv_solution, vmin=0, vmax=1)
plt.title("TV based denoising")
f.add_subplot(1, 3, 3)
plt.gray()
plt.imshow(tgv_solution, vmin=0, vmax=1)
plt.title("TGV based denoising")
plt.axis('off')
plt.show()


