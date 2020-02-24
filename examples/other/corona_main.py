# description
# description
# description

import numpy as np
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt


from recon.math.operator.mri_dft import MriDft

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
sigma = 0.005
n = sigma*max(abs(g))*np.random.uniform(0,1, g.shape[0])
f = g + n


# Plot Zero-filling reconstructions
fig, axs = plt.subplots(2, 2)

#set(figure,'defaulttextinterpreter','latex');
axs[0, 0].imshow(gt)
axs[0, 0].set_title('Groundtruth')

axs[0, 1].imshow(samp/np.max(samp))
axs[0, 1].set_title('Undersampling matrix ')

axs[1, 0].imshow(np.reshape(abs(SF.T*g), gt.shape).T, vmin=0, vmax=np.max(gt))
axs[1, 0].set_title('Zero-filled recon of n.f. data')

axs[1, 1].imshow(np.reshape(abs(SF.T*f), gt.shape).T, vmin=0, vmax=np.max(gt))
axs[1, 1].set_title('Zero-filled recon of n data')

plt.show()

print("test")


