import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from recon.reconstruction import PdSmooth, PdSmoothBregman

data_import_path = "./data/"
data_output_path = data_import_path+"output/"

img = misc.ascent()
img = img/np.max(img)
gt = img


plt.gray()
plt.imshow(img)
plt.axis('off')
plt.savefig(data_output_path+'2d_smoothing_gt.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


sigma = 0.2
n = sigma*np.max(abs(gt.ravel()))*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n
plt.gray()
plt.imshow(noise_img)
plt.axis('off')
plt.savefig(data_output_path+'2d_smoothing_noisy.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()

# TV smoothing small alpha
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=0.2, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
plt.gray()
plt.imshow(u0, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'2d_smoothing_tv.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


# Tikhonov smoothing
tikh_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tikhonov', alpha=2.6, tau=0.1346)
u0_tik = tikh_smoothing.solve(data=noise_img, maxiter=450, tol=5*10**(-6))
plt.gray()
plt.imshow(u0_tik, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'2d_smoothing_tikhonov.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


# 1d comparisson with [gt, noise, tikhonov, tv]
x_min = 84
x_max = 155
y = 20
plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
plt.plot(range(x_min, x_max), u0_tik[x_min:x_max,y], color="blue", label="Tikhonov")
plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
plt.plot(range(x_min, x_max), u0[x_min:x_max,y], color="green", label="TV")
plt.legend(loc="lower left")
plt.savefig(data_output_path+'2d_smoothing_1d_comp.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()



breg_smoothing = PdSmoothBregman(domain_shape=gt.shape,
                                 reg_mode='tv',
                                 alpha=1.1,
                                 tau=0.3182,
                                 assessment=0.6 * sigma*np.max(abs(gt.ravel())) * np.sqrt(np.prod(gt.shape)) )
u0_breg = breg_smoothing.solve(data=noise_img, maxiter=450, tol=5*10**(-6))
plt.gray()
plt.imshow(u0_breg, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'2d_smoothing_bregman.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()

# 1d comparisson with [gt, noise, bregman_tv, tv, tikhonov]
x_min = 84
x_max = 155
y = 20
plt.plot(range(x_min, x_max), u0_tik[x_min:x_max,y], color="darkcyan", label="Tikhonov")
plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
plt.plot(range(x_min, x_max), u0[x_min:x_max,y], color="green", label="TV")
plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
plt.plot(range(x_min, x_max), u0_breg[x_min:x_max,y], color="blue", label="BregTV")
plt.legend(loc="lower left")
plt.savefig(data_output_path+'2d_smoothing_1d_comp_2.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()
