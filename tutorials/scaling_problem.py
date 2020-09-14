"""
06. Scaling Dependent Stepsize Problem
================
In some tests it has been noticed that the scaling of the
image has an influence on the regularized final result.
In this tutorial an example for the occurrence of such effects is shown.
"""

###############################################################################
# A grey image is created and viewed once in area [0, 255] and once in area [0, 1].
# The relation of the weighting between dataterm and regularization
# remains the same, but is adjusted in its absolute value too.
import numpy as np
from recon.interfaces import Smoothing
import matplotlib.pyplot as plt
from pylops import Gradient

# build image
size, small_size = 256, 200
scaled_image = np.reshape(np.array([(x/size) for x in range(size)]*size), (size, size))
scaled_image[28:small_size+28, 28:small_size+28] = \
    np.reshape(np.array([(1-x/small_size)for x in range(small_size)]*small_size), (small_size, small_size))
scaled_image /= np.max(scaled_image)

assert np.all([0 <= np.min(scaled_image), np.max(scaled_image) == 1])

unscaled_image = scaled_image * 255

sigma = 0.2  # the percentage portion standard deviation for normal (Gaussian) distribution.

noise_scaled_image = scaled_image + np.random.normal(0, 0.2*np.max(scaled_image), size=(size, size))
noise_unscaled_image = unscaled_image + np.random.normal(0, 0.2*np.max(unscaled_image), size=(size, size))


###############################################################################
# ...

weights = [(0.001, 0.2), (1, 0.2), (0.005, 1)]

rows = ['{}'.format(row) for row in weights]

f = plt.figure(figsize=(6, 3*len(weights)))


for i, weight in enumerate(weights):
    tv_scaled_obj = Smoothing(domain_shape=scaled_image.shape, reg_mode='tv', lam=weight[0], alpha=weight[1], tau=0.3)
    scaled_tv_solution = tv_scaled_obj.solve(scaled_image, max_iter=350)

    tv_unscaled_obj = Smoothing(domain_shape=scaled_image.shape, reg_mode='tv', lam=weight[0], alpha=weight[1], tau=0.3)
    unscaled_tv_solution = tv_unscaled_obj.solve(unscaled_image, max_iter=350)


    f.add_subplot(3, 2, (i)*2+1)
    plt.gray()
    plt.axis('off')
    plt.imshow(scaled_tv_solution)
    plt.title("Scaled " + str(weight))
    f.add_subplot(3, 2, (i+1)*2)

    plt.gray()
    plt.imshow(unscaled_tv_solution)
    plt.title("Unscaled " + str(weight))
    plt.axis('off')
plt.tight_layout()
plt.show()

#############################################
# Gradient Verification
# To check there are no elemiation/condition things on the Gradient Operator:

grad = Gradient(dims=(size, size), edge=True, kind='backward')
scaled_gradient = grad * noise_scaled_image.ravel()
unscaled_gradient = grad * noise_unscaled_image.ravel()

scaled_reconstruction = np.reshape(grad / scaled_gradient, (size, size))
unscaled_reconstruction = np.reshape(grad / unscaled_gradient, (size, size))

assert abs(np.linalg.norm(scaled_reconstruction - noise_scaled_image) -
           np.linalg.norm((unscaled_reconstruction - noise_unscaled_image)/255) < 1)

scaled_reconstruction = grad.H / (grad.H * scaled_gradient)
unscaled_reconstruction = grad.H / (grad.H * unscaled_gradient)

assert(abs(np.linalg.norm(scaled_reconstruction - scaled_gradient) -
           np.linalg.norm((unscaled_reconstruction - unscaled_gradient)/255)) < 1)

#############################################
# Conclusion
# The Prox-Param tau is dependent on input scale.
# Therefore the calc method must be adapted in future versions.