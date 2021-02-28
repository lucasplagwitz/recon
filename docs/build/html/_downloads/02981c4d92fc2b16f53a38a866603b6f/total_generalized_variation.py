"""
04. Total Generalized Variation
===============================
We take a step deeper into total-variation-based regularization.

We focus on concepts from different papers.
Mainly we use for numerical access:
    Knoll, Bredis, Pock: Second Order Total Generalized Variation (TGV) for MRI

"""

###############################################################################
# The first order total variation got some stair-casing problems.
# See the following denoising example with the TV regularization.
import numpy as np
import matplotlib.pyplot as plt

from recon.utils import psnr
from recon.utils.images import two_smooth_squares
from recon.interfaces import Smoothing, SmoothBregman

image = two_smooth_squares(256, 128)
noise_image = image + np.random.normal(0, 0.2*np.max(image), size=image.shape)

tv_denoising = Smoothing(domain_shape=image.shape, reg_mode='tv', lam=0.3, alpha=0.1, tau='calc')
tv_solution = tv_denoising.solve(noise_image, max_iter=2000, tol=1e-4)

f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.gray()
plt.axis('off')
plt.imshow(noise_image, vmin=0, vmax=np.max(image))
plt.title("Noisy")
f.add_subplot(1, 2, 2)
plt.gray()
plt.imshow(tv_solution, vmin=0, vmax=np.max(image))
plt.title("TV based denoising")
plt.axis('off')
plt.show()

###############################################################################
# To avoid the strong stair-casing effects, we introduce the total generalized variation (TGV).
# At this point there is no interface for second order TV. We implement it direct with an
# adapted Primal-Dual algorithm.

from recon.solver.pd_hgm_tgv import PdHgmTGV

# TGV smoothing small alpha
alpha = (0.3, 0.6)
solver = PdHgmTGV(alpha=alpha, lam=0.9)
tgv_solution = np.reshape(solver.solve(noise_image), image.shape)

f = plt.figure(figsize=(9, 3))
f.add_subplot(1, 3, 1)
plt.gray()
plt.axis('off')
plt.imshow(image, vmin=0, vmax=np.max(image))
plt.title("Original")
f.add_subplot(1, 3, 2)
plt.gray()
plt.axis('off')
plt.imshow(tv_solution, vmin=0, vmax=np.max(image))
plt.title("TV based denoising")
f.add_subplot(1, 3, 3)
plt.gray()
plt.imshow(tgv_solution, vmin=0, vmax=np.max(image))
plt.title("TGV based denoising")
plt.axis('off')
plt.show()


###############################################################################
# Since TGV also represents a convex functional, it can also be extended by Bregman.
# Maybe there will be an interface for this in the future.

plot_iteration = False
lam = 0.3
assessment = 0.2 * np.max(image) * np.sqrt(np.prod(noise_image.shape))
pk = np.zeros(image.shape)
pk = pk.ravel()
i = 0

u = np.zeros(image.shape)
while True:
    print("current norm error: " + str(np.linalg.norm(u.ravel() - noise_image.ravel(), 2)))
    print("runs till norm <: " + str(assessment))

    solver = PdHgmTGV(alpha=alpha, lam=lam, mode='tgv', pk=pk)

    u_new = np.reshape(solver.solve(noise_image), image.shape)

    if np.linalg.norm(u_new.ravel() - noise_image.ravel(), 2) < assessment:
        break

    u = u_new
    pk = pk - lam / alpha[0] * (u.ravel() - noise_image.ravel())
    i = i + 1

    if plot_iteration:
        plt.gray()
        plt.imshow(u)
        plt.axis('off')
        plt.savefig('Bregman_TGV_iter' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

###############################################################################
# Compare it to normal BTV.

breg_smoothing = SmoothBregman(domain_shape=image.shape,
                               reg_mode='tv',
                               alpha=1,
                               lam=0.5,
                               tau='calc',
                               plot_iteration=False,
                               assessment=assessment)

u_breg = breg_smoothing.solve(data=noise_image, max_iter=2000, tol=1e-4)


f = plt.figure(figsize=(9, 3))
f.add_subplot(1, 3, 1)
plt.gray()
plt.axis('off')
plt.imshow(image, vmin=0, vmax=np.max(image))
plt.title("Original")
f.add_subplot(1, 3, 2)
plt.gray()
plt.axis('off')
plt.imshow(np.reshape(u_breg, image.shape), vmin=0, vmax=np.max(image))
plt.title("BTV ")
f.add_subplot(1, 3, 3)
plt.gray()
plt.imshow(np.reshape(u_new, image.shape), vmin=0, vmax=np.max(image))
plt.title("BTGV")
plt.axis('off')
plt.show()

print("TV-PSNR: "+str(psnr(image, tv_solution)))
print("TGV-PSNR: "+str(psnr(image, tgv_solution)))
print("BTV-PSNR: "+str(psnr(image, u_breg)))
print("BTGV-PSNR: "+str(psnr(image, u_new)))
