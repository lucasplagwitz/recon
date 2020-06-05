import numpy as np
import matplotlib.pyplot as plt

from recon.math.terms import Dataterm, Projection, DatatermRecBregman
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest

from recon.segmentation.tv_pdghm import multi_class_segmentation

import pylops
import nibabel as nib

plt.close('all')

data_import_path = "./../data/nifti/"
data_output_path = "./../data/nifti/output/"

img = nib.load(data_import_path+"PAC2018_0001.nii")
d = np.array(img.dataobj)[20:80, 20:82, 30:70]
d = d/np.max(d)
gt = d


classes = [0, 0.2, 0.4, 0.7]


result, _ = multi_class_segmentation(gt, classes=classes, beta=0.001)


plt.Figure()
plt.imshow(result[:,:,35])
plt.xlabel('TV based segmentation')
plt.axis('off')
plt.savefig(data_output_path+"plain_segmentation.png")
plt.close()

new_image = nib.Nifti1Image(result, affine=np.eye(4))

new_image.to_filename(data_output_path+'plain_segmentation.nii')