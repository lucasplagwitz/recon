# required nilearn
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces import Recon, ReconBregman
import numpy as np
from nilearn import datasets
import pylops
import nibabel as nib

plt.close('all')

data_import_path = "./data/"
data_output_path = data_import_path+"output/"

n_subjects = 1
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)

img = nib.load(oasis_dataset.gray_matter_maps[0])
d = np.array(img.dataobj) # [20:80,20:82, 50:55]
d = d/np.max(d)
gt = d

dx, dy, dz = 0.005, 5, 3
nx, ny, nz = d.shape
print(d.shape)

t = np.arange(nx)*dx
x = np.arange(ny)*dy
y = np.arange(nz)*dz
f0 = 10
nfft = 2**7
nfftk = 2**7

FFTop = pylops.signalprocessing.FFTND(dims=(nx, ny, nz),
                                      nffts=(nfft, nfftk, nfftk),
                                      sampling=(dx, dy, dz))
D = FFTop*d.flatten()

dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(nx, ny, nz)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d[:, :, nz//2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                        axes=1)[:20, :, nfftk//2]),
                 cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv[:, :, nz//2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d[:, :, nz//2]-dinv[:, :, nz//2]
                 , vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path + "recon_plain.png")
plt.close(fig)

new_image = nib.Nifti1Image(dinv, affine=np.eye(4))
new_image.to_filename(data_output_path+'plain_recon.nii')
new_image = nib.Nifti1Image(np.abs(d-dinv), affine=np.eye(4))
new_image.to_filename(data_output_path+'plain_recon_error.nii')

# Gaussian noise #
##################
D = FFTop*d.flatten()
sigma = 0.003
n = sigma*np.max(np.abs(D))*np.random.normal(size=D.shape[0])
D = D + n

# Adjoint = inverse for FFT
dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(d.shape)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d[:, :, nz//2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                        axes=1)[:20, :, nfftk//2]),
                 cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv[:, :, nz//2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d[:, :, nz//2]-dinv[:, :, nz//2]
                 , vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path+"recon_noise.png")
plt.close(fig)

noisy_img = dinv
new_image = nib.Nifti1Image(dinv, affine=np.eye(4))
new_image.to_filename(data_output_path+'noise_recon.nii')
new_image = nib.Nifti1Image(np.abs(d-dinv), affine=np.eye(4))
new_image.to_filename(data_output_path+'noise_recon_error.nii')



# Gaussian noise - TV regularised Reconstruction #
##################################################
tv_recon = Recon(O=FFTop,
                 domain_shape=d.shape,
                 reg_mode='tv',
                 alpha=0.1,
                 tau=2.0)

u = np.real(tv_recon.solve(D))

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                            axes=1)[:20, :, nfftk // 2]),
                     cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(u[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d[:, :, nz // 2] - u[:, :, nz // 2]
                     , vmin=np.min(d), vmax=np.max(d), cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path + "pylops_fft_3d_tv_recon.png")
plt.close(fig)

new_image = nib.Nifti1Image(u, affine=np.eye(4))
new_image.to_filename(data_output_path+'tv_recon.nii')


# Bregman reconstruction #
##########################
# too long at this point
bregman = False
if bregman:
    tv_recon = ReconBregman(O=FFTop,
                            domain_shape=d.shape,
                            reg_mode='tv',
                            tau = 0.05,
                            alpha=0.15,
                            assessment=10*sigma*np.max(abs(gt.ravel()))*np.sqrt(np.prod(gt.shape)))

    u = np.real(tv_recon.solve(D))

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0][0].imshow(d[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[0][0].set_title('Signal')
    axs[0][0].axis('tight')
    axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                                axes=1)[:20, :, nfftk // 2]),
                         cmap='seismic')
    axs[0][1].set_title('Fourier Transform')
    axs[0][1].axis('tight')
    axs[1][0].imshow(u[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[1][0].set_title('Inverted')
    axs[1][0].axis('tight')
    axs[1][1].imshow(d[:, :, nz // 2] - u[:, :, nz // 2]
                         , vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[1][1].set_title('Error')
    axs[1][1].axis('tight')
    fig.tight_layout()

    plt.savefig(data_output_path + "bregman_recon.png")
    plt.close(fig)

    new_image = nib.Nifti1Image(u, affine=np.eye(4))
    new_image.to_filename(data_output_path + 'bregman_recon.nii')

