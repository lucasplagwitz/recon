import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from recon.solver.pd_hgm_tgv import PdHgmTGV

data_import_path = "./data/"
data_output_path = data_import_path+"tgv/"

img = misc.ascent()[:256, :256]
img = img/np.max(img)
gt = img

def draw_images(img, name, vmin=0, vmax=1):
    plt.gray()
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(data_output_path + name, bbox_inches='tight', pad_inches=0)
    plt.close()

draw_images(gt, 'gt.png', vmin=0, vmax=np.max(gt))

# create noisy image
sigma = 0.05
n = sigma*np.max(abs(gt.ravel()))*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n
draw_images(noise_img, 'noisy.png', vmin=0, vmax=np.max(gt))


# TGV smoothing small alpha
alpha = (1, 1)
solver = PdHgmTGV(alpha=alpha)
u_tgv = np.reshape(solver.solve(noise_img), (256, 256))
draw_images(u_tgv, 'tgv.png', vmin=0, vmax=np.max(gt))