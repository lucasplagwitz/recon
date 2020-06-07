# RECON
A python-based research toolbox for reconstruction and segmentation to solve Inverse Problems.

## Reconstruction
In terms of Inverse Problems one is intrested in the reason 
<img src="https://render.githubusercontent.com/render/math?math=\Large u">
of measurment data 
<img src="https://render.githubusercontent.com/render/math?math=\Large f">
respect to some forward map 
<img src="https://render.githubusercontent.com/render/math?math=\Large A">.
Due to the fact of measurement inaccuracies, regularization terms 
<img src="https://render.githubusercontent.com/render/math?math=\Large J">
are added and the optimization problem is maintained as
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\Large \argmin_u ||Au - f|| %2B \alpha J(u)">
 <p/>
 
 ```python
 from recon.reconstruction import PdRecon
 import pylops
 
 FFTop = pylops.signalprocessing.FFT(dims=(nt, nx), dir=0, nfft=nfft, sampling=dt)
 D = FFTop*d.flatten()
 tv_recon = PdRecon(O=FFTop, domain_shape=d.shape, reg_mode='tv', alpha=2.0)

u = tv_recon.solve(D, maxiter=350, tol=10**(-4))
 ```
 
<table>
  <tr>
    <td>Noisy Data</td><td>TV Regularized</td>
  </tr>
  <tr>
    <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/noise_recon.gif" alt="" width="300"></td>
      <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/tv_recon.gif" alt="" width="300"> 
    </td>
  </tr>
 </table>

## Smoothing
Image Smoothing is a special case of regularized reconstruction.
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\Large \argmin_u ||u - f|| %2B \lambda J(u)">
 <p/>
 
  ```python
from scipy import misc
from recon.reconstruction import PdSmooth

img = misc.ascent()
gt = img/np.max(img)
sigma = 0.2
n = sigma*np.max(gt.ravel()*np.random.uniform(-1,1, gt.shape)
noise_img = gt + n
 
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=0.2, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=10**(-4))
 ```
 
<table>
  <tr>
    <td>Noisy Image</td><td>Tikhonov</td><td>TV</td><td>Bregman-TV</td>
  </tr>
  <tr>
    <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/2d_smoothing_noisy.png" alt="" width="200"></td>
    <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/2d_smoothing_tikhonov.png" alt="" width="200"></td>
    <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/2d_smoothing_tv.png" alt="" width="200"></td>
    <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/2d_smoothing_bregman.png" alt="" width="200"></td>
    </td>
  </tr>
 </table>
 <p align="center">
 <img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/2d_smoothing_1d_comp_2.png" alt="" width="400">
  </p>

## Segmentation
Some segmentation methods are implemented as part of regularization approaches and performance measurements.
  ```python
from recon.segmentation.tv_pdghm import multi_class_segmentation
import nibabel as nib

img = nib.load("file.nii")
d = np.array(img.dataobj)
gt = d/np.max(d)
classes = [0, 0.2, 0.4, 0.7]

result, _ = multi_class_segmentation(gt, classes=classes, beta=0.001)
 ```
<table>
  <tr>
    <td>Image</td><td>Segmentation</td>
  </tr>
  <tr>
    <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/plain_recon.gif" alt="" width="300"></td>
      <td><img src="https://github.com/lucasplagwitz/recon/blob/pylops_support/examples/demo/plain_segmentation.gif" alt="" width="300"> 
    </td>
  </tr>
 </table>

  
  ## References
  1. The Repo based on [Enhancing joint reconstruction and segmentation with non-convex Bregman iteration](https://iopscience.iop.org/article/10.1088/1361-6420/ab0b77/pdf) - Veronica Corona et al 2019 Inverse Problems 35, and their code on [GitHub](https://github.com/veronicacorona/JointReconstructionSegmentation).
  1. To outsource operator handling [pylops](https://github.com/equinor/pylops) package is used.
