# description
# description
# description

import numpy as np
from scipy import sparse
from scipy.io import loadmat


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
#S = S[samp[:],:] # ?
nz = np.count_nonzero(samp)/samp_values  # undersampling rate
#SF = np.dot(S, F);
#g =np.dot(SF,gt);

print("test")


