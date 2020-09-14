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


from recon.interfaces.satv import SATV
from recon.interfaces import Smooth, SmoothBregman

data_import_path = "./data/"
data_output_path = data_import_path+"output/"

img = misc.ascent()
img = img/np.max(img)
gt = img[250:480, 220:450]
gt[:20, :20] = 1
sigma = 0.05
n = sigma*np.max(abs(gt.ravel()))*np.random.normal(size=gt.shape)
#n = n - (n<0).astype(int) * 0.02 + (n>=0).astype(int)*0.02
#n[:-50,:-50] = 0
noise_img = gt + n

def draw_images(a, name, error=None ,max=np.max(gt)):
        plt.gray()
        plt.imshow(a, vmin=np.min(gt), vmax=max)
        plt.axis('off')
        if error !=  None:
                plt.title("NE: {:2f}".format(error))
        plt.savefig(data_output_path + name, bbox_inches='tight', pad_inches=0)
        plt.close()

draw_images(noise_img, "noise.png", np.linalg.norm(gt - noise_img, 1))




# only right sided TV smoothing for principle
alpha = 0.2
#alpha[:,:gt.shape[1]//2] = 0
tv_smoothing = Smooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha)
u0 = tv_smoothing.solve(data=noise_img, tol=5*10**(-4))
draw_images(u0, "normal_tv_regularization.png", np.linalg.norm(gt - u0, 1))


tv_smoothing = SmoothBregman(domain_shape=gt.shape, reg_mode='tv', alpha=0.6,
                             assessment=sigma*np.max(abs(gt.ravel())) * np.sqrt(np.prod(gt.shape)))
#u0 = tv_smoothing.solve(data=noise_img, tol=5*10**(-6), max_iter=450)
#draw_images(u0, "bregman_tv_regularization.png", np.linalg.norm(gt - u0, 1))



alpha = np.ones(gt.shape)*0.6
tv_smoothing = SATV(domain_shape=gt.shape, reg_mode='tv', alpha=alpha,
                    data_output_path=data_output_path, noise_sigma=sigma,
                    assessment=sigma*np.max(abs(gt.ravel())) * np.sqrt(np.prod(gt.shape)))
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "normal_tv_regularization_mtv.png", np.linalg.norm(gt - u0, 1))
