import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces import Segmentation

from nilearn import datasets
import nibabel as nib

plt.close('all')

data_import_path = "./data/"
data_output_path = "./data/output/"

n_subjects = 1
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)

img = nib.load(oasis_dataset.gray_matter_maps[0])
d = np.array(img.dataobj) #[20:80, 20:82, 30:70]
d = d/np.max(d)
gt = d


classes = [0, 0.2, 0.4, 0.7]

segmentation = Segmentation(gt.shape, classes=classes, alpha=0.1, tau=1.7)
result, _ = segmentation.solve(gt)

plt.Figure()
plt.imshow(result[:,:,35])
plt.xlabel('TV based segmentation')
plt.axis('off')
plt.savefig(data_output_path+"3d_nifti_segmentation.png")
plt.close()

new_image = nib.Nifti1Image(result, affine=np.eye(4))

new_image.to_filename(data_output_path+'3d_nifti_segmentation.nii')