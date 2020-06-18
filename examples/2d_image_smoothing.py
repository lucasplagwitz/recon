import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from recon.interfaces import Smooth, SmoothBregman

data_import_path = "./data/"
data_output_path = data_import_path+"output/"

img = misc.ascent()
img = img/np.max(img)
gt = img

def draw_images(img, name, vmin=0, vmax=1):
    plt.gray()
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(data_output_path + name, bbox_inches='tight', pad_inches=0)
    plt.close()

draw_images(gt, '2d_smoothing_gt.png', vmin=0, vmax=np.max(gt))

# create noisy image
sigma = 0.2
n = sigma*np.max(abs(gt.ravel()))*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n
draw_images(noise_img, '2d_smoothing_noisy.png', vmin=0, vmax=np.max(gt))


# TV smoothing small alpha
tv_smoothing = Smooth(domain_shape=gt.shape, reg_mode='tv', alpha=0.2, tau=2.3335)
u_tv = tv_smoothing.solve(data=noise_img, max_iter=150, tol=5*10**(-4))
draw_images(u_tv, '2d_smoothing_tv.png', vmin=0, vmax=np.max(gt))

# Tikhonov smoothing
tikh_smoothing = Smooth(domain_shape=gt.shape, reg_mode='tikhonov', alpha=1.99, tau=0.01335)
u_tik = tikh_smoothing.solve(data=noise_img, max_iter=150, tol=5*10**(-6))
draw_images(u_tik, '2d_smoothing_tikhonov.png', vmin=0, vmax=np.max(gt))


# 1d comparisson with [gt, noise, tikhonov, tv]
x_min = 84
x_max = 155
y = 20
plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
plt.plot(range(x_min, x_max), u_tik[x_min:x_max,y], color="blue", label="Tikhonov")
plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
plt.plot(range(x_min, x_max), u_tv[x_min:x_max,y], color="green", label="TV")
plt.legend(loc="lower left")
plt.savefig(data_output_path+'2d_smoothing_1d_comp.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()



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
