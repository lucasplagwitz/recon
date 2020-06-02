import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import sparse

from recon.math.operator.mri_dft import MriDft
from recon.math.terms import Dataterm, Projection
from recon.math.terms.dataterm_linear_rec_bregman import DatatermLinearRecBregman
from recon.math.terms.dataterm_linear import DatatermLinear
from recon.math.operator.first_derivative import FirstDerivative
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest

data_import_path = "./../data/smoothing_images/"
data_output_path = data_import_path+"output/"

img = misc.ascent()
img = img/np.max(img)
gt = img


plt.gray()
plt.imshow(img)
plt.axis('off')
plt.savefig(data_output_path+'gt.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


sigma = 0.2
n = sigma*np.max(abs(gt.ravel()))*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n
plt.gray()
plt.imshow(noise_img)
plt.axis('off')
plt.savefig(data_output_path+'noisy.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


# Gradient  operator
ex = np.ones((gt.shape[1],1))
ey = np.ones((1, gt.shape[0]))
dx = sparse.diags([1, -1], [0, 1], shape=(gt.shape[1], gt.shape[1])).tocsr()
dx[gt.shape[1]-1, :] = 0
dy = sparse.diags([-1, 1], [0, 1], shape=(gt.shape[0], gt.shape[0])).tocsr()
dy[gt.shape[0]-1, :] = 0

grad = sparse.vstack((sparse.kron(dx, sparse.eye(gt.shape[0]).tocsr()),
                      sparse.kron(sparse.eye(gt.shape[1]).tocsr(), dy)))


# TV smoothing small alpha
alpha0=0.2
K = alpha0*grad

norm = normest(K)
sigma0 = 0.99 / norm
tau0 = sigma0

G = DatatermLinear()
F_star = Projection(gt.shape)
solver = PdHgm(K, F_star, G)
G.set_proxparam(tau0)
F_star.set_proxparam(sigma0)
solver.maxiter = 450
solver.tol = 5*10**(-4)

G.set_proxdata(noise_img.ravel())
solver.solve()

u0 = np.reshape(solver.var['x'], gt.shape)

plt.gray()
plt.imshow(u0, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'tv_filter.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()



# Tikhonov smoothing big alpha
alpha0=2.6
K = alpha0 * grad #sparse.eye(grad.shape[0], grad.shape[1])

norm = normest(K)
sigma0 = 0.99 / norm
tau0 = sigma0

G = DatatermLinear()
F_star = DatatermLinear()
F_star.set_proxdata(0)
solver = PdHgm(K, F_star, G)
G.set_proxparam(tau0)
F_star.set_proxparam(1)
solver.maxiter = 250
solver.tol = 5*10**(-4)
G.set_proxdata(noise_img.ravel())
solver.solve()

u0_tik= np.reshape(solver.var['x'], gt.shape)

plt.gray()
plt.imshow(u0_tik, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'tikhonov_filter.png', bbox_inches = 'tight', pad_inches = 0)

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
plt.savefig(data_output_path+'1d_comp.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()



# Bregman TV smoothing
alpha01=1.1  # regularisation parameter
K01 = alpha01*grad
norm = normest(K)
sigma0 = 0.99 / norm
tau0 = sigma0
plot_iteration = False

pk = np.zeros(gt.shape)
pk = pk.T.ravel()
plt.Figure()
ulast = np.zeros(gt.shape)
u01=ulast
i=0


#while np.linalg.norm(SF * u01.ravel()-f.ravel(), ord=2) > 0.005 * np.max(abs(g)) * np.sqrt(np.prod(f.shape)):
while np.linalg.norm(u01.ravel()-noise_img.ravel()) > 5/9*sigma * np.sqrt(np.prod(gt.shape)):
    print(np.linalg.norm(u01.ravel()-noise_img.ravel()))
    print(sigma * np.sqrt(np.prod(gt.shape)))
    #print(np.sqrt(np.sum(abs(SF * u01.ravel() - f.ravel())**2)))
    ulast = u01

    G = DatatermLinearRecBregman()
    F_star = Projection(gt.shape)

    solver = PdHgm(K01, F_star, G)
    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 250
    solver.tol = 5 * 10**(-4)

    G.set_proxdata(noise_img.ravel())
    G.setQ(pk)
    solver.solve()
    u01 = np.reshape(np.real(solver.var['x']), gt.shape)
    pklast = pk
    pk = pk - (1/alpha01) * (u01.ravel() -noise_img.ravel())
    i=i+1
    if plot_iteration:
        plt.gray()
        plt.imshow(u01, vmin=0, vmax=np.max(gt))
        plt.axis('off')
        #plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
        plt.savefig(data_output_path+'Bregman_reconstruction_iter'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()


# 1d comparisson with [gt, noise, bregman_tv, tv]
x_min = 84
x_max = 155
y = 20
plt.plot(range(x_min, x_max), gt[x_min:x_max,y], color="black", label="GT")
plt.plot(range(x_min, x_max), u01[x_min:x_max,y], color="blue", label="BregTV")
plt.plot(range(x_min, x_max), noise_img[x_min:x_max,y], color="red", label="Noise")
plt.plot(range(x_min, x_max), u0[x_min:x_max,y], color="green", label="TV")
plt.legend(loc="lower left")
plt.savefig(data_output_path+'1d_comp_2.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


# segmentation of [gt, noise, bregman_tv, tv]
from recon .segmentation.tv_pdghm import multi_class_segmentation

classes = [0, 0.3, 0.7, 1]
beta = 0.001
tau = 350

a, _ = multi_class_segmentation(gt, classes, beta, tau)
plt.imshow(a)
plt.axis('off')
plt.savefig(data_output_path+'gt_seg_'+str(beta)+'.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()


b, _ = multi_class_segmentation(u01, classes, beta, tau)
plt.imshow(b)
plt.axis('off')
plt.savefig(data_output_path+'breg_seg_'+str(beta)+'.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()

c, _ = multi_class_segmentation(noise_img, classes, beta, tau)
plt.imshow(c)
plt.axis('off')
plt.savefig(data_output_path+'noise_seg_'+str(beta)+'.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()

d, _ = multi_class_segmentation(u0, classes, beta, tau)
plt.imshow(d)
plt.axis('off')
plt.savefig(data_output_path+'tv_seg_'+str(beta)+'.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()

