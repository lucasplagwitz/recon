import numpy as np
import matplotlib.pyplot as plt

from recon.math.terms import Dataterm, Projection, DatatermRecBregman
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest

import pylops
import nibabel as nib

plt.close('all')

data_import_path = "./../data/nifti/"
data_output_path = "./../data/nifti/output/"

img = nib.load(data_import_path+"PAC2018_0001.nii")
d = np.array(img.dataobj)[20:80,20:82, 30:70]
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

#d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
#d = np.tile(d[:, :, np.newaxis], [1, 1, ny])


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

mse = np.sqrt(np.sum((d-dinv)**2))
print(20*np.log(1/mse))

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

mse = np.sqrt(np.sum((d-dinv)**2))
print(20*np.log(1/mse))

# Gaussian noise - TV regularised Reconstruction #
##################################################
tv_recon = False
recalc_normest = True
if tv_recon:
    #ny = 100 #2000
    #nx = 20  #20480
    #grad = pylops.FirstDerivative(nt * nx, dims=(nt, nx), dir=0, dtype='float64')
    grad = pylops.Gradient(dims=(nx, ny, nz), dtype='float64')
    alpha0=0.05
    K = alpha0*grad

    if recalc_normest:
        #norm = normest(K)
        norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
        sigma0=0.99/norm
        print(sigma0)
    else:
        sigma0= 0.0398
    tau0 = sigma0

    G = Dataterm(S=None, F=FFTop)
    F_star = Projection(d.shape, 3)
    solver = PdHgm(K, F_star, G)
    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 250
    solver.tol = 5*10**(-4)

    G.set_proxdata(D)
    solver.solve()

    u0 = np.reshape(np.real(solver.var['x']), d.shape)
#    rel_tvrec = np.linalg.norm(d - u0, 2) #/np.linalg.norm(gt)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0][0].imshow(d[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[0][0].set_title('Signal')
    axs[0][0].axis('tight')
    axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                            axes=1)[:20, :, nfftk // 2]),
                     cmap='seismic')
    axs[0][1].set_title('Fourier Transform')
    axs[0][1].axis('tight')
    axs[1][0].imshow(u0[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[1][0].set_title('Inverted')
    axs[1][0].axis('tight')
    axs[1][1].imshow(d[:, :, nz // 2] - u0[:, :, nz // 2]
                     , vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[1][1].set_title('Error')
    axs[1][1].axis('tight')
    fig.tight_layout()

    plt.savefig(data_output_path + "pylops_fft_3d_tv_recon.png")
    plt.close(fig)

    new_image = nib.Nifti1Image(u0, affine=np.eye(4))

    new_image.to_filename(data_output_path+'tv_recon.nii')


img = nib.load(data_output_path+'tv_recon.nii')
dinv = np.array(img.dataobj)

new_image = nib.Nifti1Image(np.abs(d-dinv), affine=np.eye(4))

new_image.to_filename(data_output_path+'tv_recon_error.nii')

mse = np.sqrt(np.sum((d-dinv)**2))
print(20*np.log(1/mse))

bregman_recon = False
if bregman_recon:
    # Bregman TV smoothing
    alpha01 = 1.1  # regularisation parameter
    grad = pylops.Gradient(dims=(nx, ny, nz), dtype='float64')
    K = alpha01 * grad
    if recalc_normest:
        # norm = normest(K)
        norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
        sigma0 = 0.99 / norm
        print(sigma0)
    else:
        sigma0 = 0.0398
    tau0 = sigma0
    plot_iteration = True

    pk = np.zeros(gt.shape)
    pk = pk.T.ravel()
    plt.Figure()
    ulast = np.zeros(gt.shape)
    u01 = ulast
    i = 0

    # while np.linalg.norm(SF * u01.ravel()-f.ravel(), ord=2) > 0.005 * np.max(abs(g)) * np.sqrt(np.prod(D.shape)):
    while i < 13:#np.linalg.norm(u01.ravel() - noisy_img.ravel()) > np.max(np.abs(D)) * sigma * np.sqrt(np.prod(D.shape)):
        print(np.linalg.norm(u01.ravel() - noisy_img.ravel()))
        print(np.max(np.abs(D)) * sigma * np.sqrt(np.prod(D.shape)))
        ulast = u01

        G = DatatermRecBregman(S=None, F=FFTop)
        F_star = Projection(gt.shape, 3)

        solver = PdHgm(K, F_star, G)
        G.set_proxparam(tau0)
        F_star.set_proxparam(sigma0)
        solver.maxiter = 250
        solver.tol = 5 * 10 ** (-4)

        G.set_proxdata(D)
        G.setP(pk)
        solver.solve()
        u01 = np.reshape(np.real(solver.var['x']), gt.shape)
        pklast = pk
        pk = pk - (1 / alpha01) * (np.real(FFTop.H * (FFTop * u01.ravel() - D)))
        i = i + 1

        RRE_breg = np.linalg.norm(gt.ravel() - u01.ravel(), 2)
        if plot_iteration:
            plt.imshow(u01[:,:,nz//2], vmin=0, vmax=np.max(gt))
            plt.axis('off')
            plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)

            plt.savefig(data_output_path + 'Bregman_reconstruction_iter' + str(i) + '.png', bbox_inches='tight',
                        pad_inches=0)
            plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0][0].imshow(d[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[0][0].set_title('Signal')
    axs[0][0].axis('tight')
    axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                            axes=1)[:20, :, nfftk // 2]),
                     cmap='seismic')
    axs[0][1].set_title('Fourier Transform')
    axs[0][1].axis('tight')
    axs[1][0].imshow(u01[:, :, nz // 2], vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[1][0].set_title('Inverted')
    axs[1][0].axis('tight')
    axs[1][1].imshow(d[:, :, nz // 2] - u01[:, :, nz // 2]
                     , vmin=np.min(d), vmax=np.max(d), cmap='seismic')
    axs[1][1].set_title('Error')
    axs[1][1].axis('tight')
    fig.tight_layout()

    plt.savefig(data_output_path + "bregman_recon.png")
    plt.close(fig)

    new_image = nib.Nifti1Image(u01, affine=np.eye(4))

    new_image.to_filename(data_output_path + 'bregman_recon.nii')


img = nib.load(data_output_path+'bregman_recon.nii')
dinv = np.array(img.dataobj)

new_image = nib.Nifti1Image(np.abs(d-dinv), affine=np.eye(4))

new_image.to_filename(data_output_path+'bregman_recon_error.nii')

mse = np.sqrt(np.sum((d-dinv)**2))
print(20*np.log(1/mse))
