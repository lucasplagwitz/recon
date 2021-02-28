"""
08. Scaling Dependent Stepsize Problem
======================================
In earlier tests it was noticed that the size of the weighting parameters
has an effect on the solution while keeping the ratio constant.
Meanwhile the problem has been identified as a too small number of iterations.
Overall, the default parameters have been adjusted, but in this example,
we will briefly show what the effect looks like.
"""

###############################################################################
# A grey image is created and viewed in an area [0, 1].
# The relation of the weighting between dataterm and regularization
# remains the same, but is adjusted in its absolute value too.
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces import Smoothing
from recon.utils.images import two_smooth_squares

scaled_image = two_smooth_squares(256, 200)

sigma = 0.3  # the percentage portion standard deviation for normal (Gaussian) distribution.

noise_image = scaled_image + np.random.normal(0, sigma*np.max(scaled_image), size=scaled_image.shape)

weights = [(0.2, 0.2), (1, 1), (2, 2)]

rows = ['{}'.format(row) for row in weights]

f = plt.figure(figsize=(6, 3*len(weights)))

for i, weight in enumerate(weights):
    tv_scaled_obj = Smoothing(domain_shape=scaled_image.shape,
                              reg_mode='tv',
                              lam=weight[0],
                              alpha=weight[1],
                              tau="calc")
    scaled_tv_solution = tv_scaled_obj.solve(noise_image, max_iter=5550, tol=1e-4)

    tv_unscaled_obj = Smoothing(domain_shape=scaled_image.shape,
                                reg_mode='tv',
                                lam=weight[0],
                                alpha=weight[1],
                                tau="calc")
    unscaled_tv_solution = tv_unscaled_obj.solve(noise_image, max_iter=550, tol=1e-4)

    f.add_subplot(3, 2, i*2+1)
    plt.gray()
    plt.axis('off')
    plt.imshow(scaled_tv_solution)
    plt.title("Long-Run: weight " + str(weight))
    f.add_subplot(3, 2, (i+1)*2)

    plt.gray()
    plt.imshow(unscaled_tv_solution)
    plt.title("Short-Run: weight " + str(weight))
    plt.axis('off')
plt.tight_layout()
plt.show()

#############################################
# Conclusion
# Be careful with max_iter and tol parameter
# or with the interpretation of result if the number of iteration is too small.
