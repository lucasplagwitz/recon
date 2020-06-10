"""
################
# EXPERIMENTAL #
################

First try of some new thoughts:
Local Regularization
In some areas more TV than in others.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from recon.reconstruction import PdSmooth
from pylops import FirstDerivative, Smoothing2D

data_import_path = "./data/"
data_output_path = data_import_path+"output/"

img = misc.ascent()
img = img/np.max(img)
gt = img
sigma = 0.2
n = sigma*np.max(abs(gt.ravel()))*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n

def draw_images(a, name):
        plt.gray()
        plt.imshow(a)
        plt.axis('off')
        plt.savefig(data_output_path + name, bbox_inches='tight', pad_inches=0)
        plt.close()


# only right sided TV smoothing for principle
alpha = np.ones(gt.shape)*0.4
alpha[:,:gt.shape[1]//2] = 0
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha, tau=0.875)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "splitted_regularization.png")


# Gradient based local regularization - decreasing Gradient influence
Sop = Smoothing2D(nsmooth=[3, 3], dims=gt.shape, dtype='float64')

alpha = np.ones(gt.shape)*0.4 \
        - 2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=0)*gt.ravel(), gt.shape)) \
        - 2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=1)*gt.ravel(), gt.shape))
alpha = np.clip(alpha,0,1)
alpha = alpha/np.mean(alpha)*0.6
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha) #, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "splitted_regularization_1.png")
draw_images(alpha, "splitted_regularization_1alpha.png")


# only left sided TV smoothing - increasing Gradient influence
Sop = Smoothing2D(nsmooth=[3, 3], dims=gt.shape, dtype='float64')

alpha = np.ones(gt.shape)*0.4 \
        + 2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=0)*gt.ravel(), gt.shape)) \
        + 2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=1)*gt.ravel(), gt.shape))
alpha = alpha/np.mean(alpha)*0.6
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha) #, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "splitted_regularization_2.png")
draw_images(alpha, "splitted_regularization_2alpha.png")

"""
# some not implemented alternative algorithms...
alpha = np.ones(gt.shape)*0.1
tv_smoothing = PdSmoothSPTV(domain_shape=gt.shape,
                            reg_mode='tv',
                            alpha=alpha,
                            noise_sigma=np.sqrt(sigma),
                            data_output_path=data_output_path) #, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
plt.gray()
plt.imshow(u0, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'2d_local_smoothing_tv.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()
"""