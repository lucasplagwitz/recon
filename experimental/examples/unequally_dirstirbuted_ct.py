import numpy as np
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
import imageio

from pylops import Gradient

from recon.interfaces import ReconBregman, Recon
from experimental.interfaces.recon_satv import ReconSATV
from experimental.interfaces.recon_satv_direct import ReconSATVDirect
from experimental.operator.ct_radon import CtRt

# load content
image_path = "./../../examples/data/gt.png"
gt = imageio.imread(image_path, as_gray=True)
gt = gt/np.max(gt)
gt = gt[:64, :64]
gt[:,:] = 0
gt[10:30, 10:30] = 0.5
gt[20:26, 20:26] = 0

for i in [40, 45, 50]:
    for j in [40, 45, 50]:
        gt[i:i+4, j:j+4] = 0.5


theta = list(np.linspace(0., 60., 60, endpoint=False)) +\
        list(np.linspace(60., 120., 60, endpoint=False)) +\
        list(np.linspace(120., 180., 60, endpoint=False))

F = CtRt(np.shape(gt),
         np.array([(np.shape(gt)[0]/2)+1, (np.shape(gt)[0]/2)+1]),
         theta=theta)


g = F * gt.ravel()

# Gaussian noise
sigma = 0.05
n = sigma*np.max(np.abs(g))*np.random.normal(size=g.shape)
f = g + n

grad = Gradient(gt.shape, edge=True, dtype='float64', kind='backward')



plt.imshow(gt)
plt.axis('off')
plt.savefig('./data/gt.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.reshape(abs(F.H*g), gt.shape), vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig('./data/recon_noise_free.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.reshape(abs(F.H*f), gt.shape), vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig('./data/recon_noise_added.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.abs(np.reshape(F.H*f, gt.shape)-gt))
plt.axis('off')
plt.colorbar()
plt.savefig('./data/noise.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.abs(np.reshape(F.H*g, gt.shape)-gt))
plt.axis('off')
plt.colorbar()
plt.savefig('./data/noise_free_noise.png', bbox_inches = 'tight', pad_inches = 0)
# non-constant gt

plt.imshow(np.reshape(F.H*n, gt.shape))
plt.axis('off')
plt.colorbar()
plt.savefig('./data/only_noise_on_zero.png', bbox_inches = 'tight', pad_inches = 0)

results = F.H*n
print("Varianz: "+str(np.std(results)))
print("Erwartungswert: "+str(sum(results) / len(results)))

"""
# SATV Direct Hintermüller K=Radon-Operator F
data_output_path = "./data/output2/"
alpha = np.ones(F.image_dim)*1.0
tv_smoothing = ReconSATVDirect(F,
                               domain_shape=gt.shape,
                               image_shape=F.image_dim,
                               reg_mode='tv', alpha=alpha,
                               data_output_path=data_output_path, noise_sigma=sigma,
                               assessment=sigma*np.max(np.abs(g))*np.sqrt(np.prod(g.shape)))
u0 = tv_smoothing.solve(data=f, maxiter=150, tol=5*10**(-5))
plt.imshow(np.reshape(u0, gt.shape))
plt.axis('off')
plt.savefig('./data/tv_regularization_satv_dirct.png', bbox_inches = 'tight', pad_inches = 0)
"""

data_output_path = "./data/output2/"
alpha = np.ones(gt.shape)*1.0
tv_smoothing = ReconSATV(domain_shape=gt.shape,
                         reg_mode='tv',
                         alpha=alpha,
                         data_output_path="./data/output3/",
                         noise_sigma=np.std(results),
                         tau=0.9,
                         assessment=np.std(results)*np.sqrt(np.prod(gt.shape)))
tv_smoothing.w = np.reshape(F.H*f, gt.shape)
u0 = tv_smoothing.solve(data=f, maxiter=150, tol=5*10**(-6))
plt.imshow(np.reshape(u0, gt.shape),vmin=0, vmax=0.5)
plt.axis('off')
plt.savefig('./data/tv_regularization_satv_schoenlieb.png', bbox_inches = 'tight', pad_inches = 0)


"""
tv_smoothing = Recon(F,
                    domain_shape=gt.shape,
                         reg_mode='tikhonov',
                         alpha=1/1000000000
                        )
u0 = tv_smoothing.solve(data=f, max_iter=450, tol=5*10**(-6))
plt.imshow(np.reshape(u0, gt.shape), vmin=0, vmax=np.max(gt))
plt.savefig('./data/tv_regularization.png', bbox_inches = 'tight', pad_inches = 0)
"""

"""
tv_smoothing = ReconBregman(O=F,
                            domain_shape=gt.shape, reg_mode='tv', alpha=0.6,
                            assessment=sigma * np.max(np.abs(g)) * np.sqrt(np.prod(f.shape)))
u0 = tv_smoothing.solve(data=f, tol=5*10**(-6), max_iter=450)
plt.imshow(u0)
plt.axis('off')
plt.colorbar()
plt.savefig('./data/tv_regularization_bregman.png', bbox_inches = 'tight', pad_inches = 0)
"""
"""
# load content
content = loadmat('./../data/brainphantom.mat')
image_path = "./../../examples/data/gt.png"
gt = imageio.imread(image_path, as_gray=True)
gt_seg = content["gt_seg"] - 1
content = loadmat('./../data/spiralsampling.mat')
samp = content["samp"]


F = CtRt(np.shape(gt), np.array([(np.shape(gt)[0]/2)+1, (np.shape(gt)[0]/2)+1]))


samp_values = np.prod(np.shape(samp))

S = sparse.eye(samp_values)
a = samp.ravel()
a = np.array(np.where(a==1)[0])
S = S.tocsr()[a,:] # ?
nz = np.count_nonzero(samp)/samp_values  # undersampling rate
SF = F
g =SF*gt.ravel()

# Gaussian noise
sigma = 0.005
n = sigma*max(abs(g))*np.random.uniform(-1,1, g.shape[0])
f = g + n


# Gradient  operator
ex = np.ones((gt.shape[1],1))
ey = np.ones((1, gt.shape[0]))
dx = sparse.diags([1, -1], [0, 1], shape=(gt.shape[1], gt.shape[1])).tocsr()
dx[gt.shape[1]-1, :] = 0
dy = sparse.diags([-1, 1], [0, 1], shape=(gt.shape[0], gt.shape[0])).tocsr()
dy[gt.shape[0]-1, :] = 0

grad = sparse.vstack((sparse.kron(dx, sparse.eye(gt.shape[0]).tocsr()),
                      sparse.kron(sparse.eye(gt.shape[1]).tocsr(), dy)))



plt.imshow(gt)
plt.axis('off')
plt.savefig('./ct/01_direct_recon/non_const/gt.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(samp/np.max(samp))
plt.axis('off')
plt.savefig('./ct/01_direct_recon/non_const/under_sampling.png', bbox_inches = 'tight', pad_inches = 0)


plt.imshow(np.reshape(abs(SF.T*g), gt.shape), vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig('./ct/01_direct_recon/non_const/recon_noise_free.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.reshape(abs(SF.T*f), gt.shape), vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig('./ct/01_direct_recon/non_const/recon_noise_added.png', bbox_inches = 'tight', pad_inches = 0)
"""