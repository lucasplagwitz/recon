# PyRegIP

A python-based toolbox for solving regularized Inverse Problems using Primal-Dual algorithms.

## Overview

* Reconstruction, Smoothing 
* class-based Segmentation
* Locally Adapted Regularization
* Bregman-Iteration


## Reconstruction
In terms of Inverse Problems one is interested in the reason 
<img src="https://render.githubusercontent.com/render/math?math=\Large u">
of measurment data 
<img src="https://render.githubusercontent.com/render/math?math=\Large f">
with regard to a forward map 
<img src="https://render.githubusercontent.com/render/math?math=\Large A">.
Due to the fact of measurement inaccuracies, regularization terms 
<img src="https://render.githubusercontent.com/render/math?math=\Large J">
are added and the optimization problem is maintained as
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\Large \argmin_u \frac{\lambda}{2}||Au - f||^2 %2B \alpha J(u)">
 <p/>
 
 ```python
 from recon.interfaces import Recon
 import pylops
 
 FFTop = pylops.signalprocessing.FFT(dims=(nt, nx), dir=0, nfft=nfft, sampling=dt)
 D = FFTop*d.flatten() + n
 tv_recon = Recon(O=FFTop, domain_shape=d.shape, reg_mode='tv', alpha=2.0)

u = tv_recon.solve(D, maxiter=350, tol=10**(-4))
 ```

## Smoothing
Image Smoothing is a special case of regularized reconstruction.
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\Large \argmin_u \frac{\lambda}{2}||u - f||^2 %2B \alpha J(u)">
 <p/>
 
  ```python
from scipy import misc
from recon.interfaces import Smoothing

img = misc.ascent()
gt = img/np.max(img)
sigma = 0.2
n = sigma*np.max(gt.ravel()*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n
 
tv_smoothing = Smoothing(domain_shape=gt.shape, reg_mode='tv', alpha=0.2, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=10**(-4))
 ```
 
 <table>
  <tr>
    <td><img src="./docs/source/tutorials/images/sphx_glr_2d_image_smoothing_001.png" alt="" width="800"></td>
 </tr>
 <tr>
    <td><img src="./docs/source/tutorials/images/sphx_glr_2d_image_smoothing_002.png" alt="" width="800"></td>
    </td>
  </tr>
 </table>
 <p align="center">
 <img src="./docs/source/tutorials/images/sphx_glr_2d_image_smoothing_003.png" alt="" width="400">
 </p>

## Segmentation
Some segmentation methods are implemented as part of regularization approaches and performance measurements.
Through a piecewise constant TV-solution, one quickly obtains a suitable segmentation.
  ```python
from recon.interfaces import Segmentation
import nibabel as nib

# segmentation of 3D nifti image
img = nib.load("file.nii")
d = np.array(img.dataobj)
gt = d/np.max(d)
classes = [0, 0.2, 0.4, 0.7]

segmentation = Segmentation(img.shape, classes=classes, alpha=0.1)
result, _ = segmentation.solve(img)
 ```

  
  ## References
  1. The Repo based on [Enhancing joint reconstruction and segmentation with non-convex Bregman iteration](https://iopscience.iop.org/article/10.1088/1361-6420/ab0b77/pdf) - Veronica Corona et al 2019 Inverse Problems 35, and their code on [GitHub](https://github.com/veronicacorona/JointReconstructionSegmentation).
  1. To outsource operator handling [PyLops](https://github.com/equinor/pylops) package is used.
