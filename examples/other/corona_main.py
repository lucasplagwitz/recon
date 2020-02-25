# description
# description
# description

import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.io import loadmat
import matplotlib.pyplot as plt


from recon.math.operator.mri_dft import MriDft
from recon.math.terms import Dataterm, Projection, DatatermRecBregman
from recon.math.pd_hgm import PdHgm

from recon.segmentation.chan_vese import chan_vese

# load content
content = loadmat('./../data/brainphantom.mat')
gt = content["gt"]
gt_seg = content["gt_seg"]
content = loadmat('./../data/spiralsampling.mat')
samp = content["samp"]


F = MriDft(np.shape(gt), np.array([(np.shape(gt)[0]/2)+1, (np.shape(gt)[0]/2)+1]))


samp_values = np.prod(np.shape(samp))

S = sparse.eye(samp_values)
a = samp.T.ravel()
a = np.array(np.where(a==1)[0])
S = S.tocsr()[a,:] # ?
nz = np.count_nonzero(samp)/samp_values  # undersampling rate
SF = S*F
g =SF*gt.T.ravel()

# Gaussian noise
sigma = 0.001
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


# Plot Zero-filling reconstructions
fig, axs = plt.subplots(2, 2)

#set(figure,'defaulttextinterpreter','latex');
axs[0, 0].imshow(gt)
axs[0, 0].set_title('Groundtruth')
axs[0, 0].axis('off')

axs[0, 1].imshow(samp/np.max(samp))
axs[0, 1].set_title('Undersampling matrix ')
axs[0, 1].axis('off')

axs[1, 0].imshow(np.reshape(abs(SF.T*g), gt.shape).T, vmin=0, vmax=np.max(gt))
axs[1, 0].set_title('Zero-filled recon of n.f. data')
axs[1, 0].axis('off')

axs[1, 1].imshow(np.reshape(abs(SF.T*f), gt.shape).T, vmin=0, vmax=np.max(gt))
axs[1, 1].set_title('Zero-filled recon of n data')
axs[1, 1].axis('off')

plt.savefig('./corona_images/beginning.png')
plt.close(fig)

# TV regularised Reconstruction
alpha0=0.2
K = alpha0*grad
#sigma0=1/np.linalg.norm(K.toarray(),2)
#tau0=1/sparse.linalg.norm(K)
sigma0= 1.7785
tau0 = sigma0

G = Dataterm(S, F)
F_star = Projection(gt.shape)
solver = PdHgm(K, F_star, G)
G.set_proxparam(tau0)
F_star.set_proxparam(sigma0)
solver.maxiter = 20
solver.tol = 5*10**(-4)

G.set_proxdata(f)
solver.solve()

u0 = np.reshape(np.real(solver.var['x']), gt.shape)
rel_tvrec = np.linalg.norm(gt - u0)/np.linalg.norm(gt)

plt.Figure()
plt.imshow(u0.T, vmin=0, vmax=np.max(gt))
plt.xlabel('TV Reconstruction, alpha=' + str(alpha0) +', RRE ='+ str(rel_tvrec))
plt.axis('off')
plt.savefig('./corona_images/tv_reconstruction.png')

# Segmentation on TV regularised Reconstruction
# currently only one levelset supported
#beta0 = 0.001  #regulatisation parameter
#c1 = 0.01; c2 = 0.3; c3 = 0.65; c4 = 0.8 # segmentation constants
#classes = [c1 c2 c3 c4]; classes = classes';

"""
vd1 = chan_vese(u0/np.max(u0)*255)

plt.Figure()
plt.imshow(vd1.T, vmin=0, vmax=np.max(gt))
plt.xlabel('TV Segmentation,')
plt.axis('off')
plt.savefig('./corona_images/tv_segmentation.png')
"""

# Bregman TV Reconstruction
alpha01=1.1  # regularisation parameter
K01 = alpha01*grad
#sigma0=1/normest(K01);tau0=1/normest(K01);
sigma0 = 0.3238
tau0 = sigma0
pk = np.zeros(gt.shape)
pk = pk.T.ravel()
plt.Figure()
ulast = np.zeros(gt.shape)
u01=ulast
i=0


while np.linalg.norm(SF * u01.ravel()-f.T.ravel()) > 0.001 * np.max(abs(g)) * np.sqrt(np.prod(f.shape)):
    print(np.linalg.norm(SF * u01.ravel()-f.T.ravel()))
    ulast = u01

    G = DatatermRecBregman(S, F)
    F_star = Projection(gt.shape)

    solver = PdHgm(K01, F_star, G)
    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 150
    solver.tol = 5 * 10**(-4)

    G.set_proxdata(f)
    G.pk=pk
    solver.solve()
    u01 = np.reshape(np.real(solver.var['x']), gt.shape)
    pklast = pk
    pk = pk - (1/alpha01) * (np.real(F.T*S.T*( SF * u01.ravel() -f)))
    #ax = fig.add_subplot(3, 3, i+1)
    #ax.imshow(u01.T, vmin=0, vmax=np.max(gt))
    #ax.axis('off')
    #i=i+1
    #if i == 9:
    #    break

#ax = fig.add_subplot(3, 3, i+1)
#ax.imshow(u01.T, vmin=0, vmax=np.max(gt))
#ax.axis('off')
fig = plt.figure()
u0B=ulast
RRE_breg=np.linalg.norm(gt - u0B, 2)/np.linalg.norm(gt)
plt.imshow(u0B.T, vmin=0, vmax=np.max(gt) )
plt.axis('off')
plt.xlabel('Bregman Reconstruction, alpha='+ str(alpha01) +', RRE =' + str(RRE_breg))

plt.savefig('./corona_images/Bregman_reconstruction.png')

print("test")


