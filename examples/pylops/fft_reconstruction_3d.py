import numpy as np
import matplotlib.pyplot as plt

from recon.math.terms import Dataterm, Projection
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest

import pylops

plt.close('all')

data_output_path = "./../data/pylops/output/"

dt, dx, dy = 0.005, 5, 3
nt, nx, ny = 30, 21, 11
t = np.arange(nt)*dt
x = np.arange(nx)*dx
y = np.arange(nx)*dy
f0 = 10
nfft = 2**6
nfftk = 2**5

d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
d = np.tile(d[:, :, np.newaxis], [1, 1, ny])

FFTop = pylops.signalprocessing.FFTND(dims=(nt, nx, ny),
                                      nffts=(nfft, nfftk, nfftk),
                                      sampling=(dt, dx, dy))
D = FFTop*d.flatten()

dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(nt, nx, ny)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d[:, :, ny//2], vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                        axes=1)[:20, :, nfftk//2]),
                 cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv[:, :, ny//2], vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d[:, :, ny//2]-dinv[:, :, ny//2],
                 vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path + "pylops_recon_3d_plain.png")
plt.close(fig)


# Gaussian noise #
##################
D = FFTop*d.flatten()
sigma = 0.05
n = sigma*np.max(np.abs(D))*np.random.normal(size=D.shape[0])
D = D + n

# Adjoint = inverse for FFT
dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(d.shape)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d[:, :, ny//2], vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                        axes=1)[:20, :, nfftk//2]),
                 cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv[:, :, ny//2], vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d[:, :, ny//2]-dinv[:, :, ny//2],
                 vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path+"pylops_fft_3d_noise.png")
plt.close(fig)


# Gaussian noise - TV regularised Reconstruction #
##################################################
tv_recon = True
recalc_normest = True
if tv_recon:
    #ny = 100 #2000
    #nx = 20  #20480
    #grad = pylops.FirstDerivative(nt * nx, dims=(nt, nx), dir=0, dtype='float64')
    grad = pylops.Gradient(dims=(ny, nx, nt), dtype='float64')
    alpha0=15.0
    K = alpha0*grad

    if recalc_normest:
        norm = normest(K)
        sigma0=0.99/norm
        print(sigma0)
    else:
        sigma0= 3.522
    tau0 = sigma0

    G = Dataterm(S=None, F=FFTop)
    F_star = Projection(d.shape, 3)
    solver = PdHgm(K, F_star, G)
    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 1500
    solver.tol = 5*10**(-12)

    G.set_proxdata(D)
    solver.solve()

    u0 = np.reshape(np.real(solver.var['x']), d.shape)
#    rel_tvrec = np.linalg.norm(d - u0, 2) #/np.linalg.norm(gt)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0][0].imshow(d[:, :, ny//2], vmin=-20, vmax=20, cmap='seismic')
    axs[0][0].set_title('Signal')
    axs[0][0].axis('tight')
    axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                            axes=1)[:20, :, nfftk // 2]),
                     cmap='seismic')
    axs[0][1].set_title('Fourier Transform')
    axs[0][1].axis('tight')
    axs[1][0].imshow(u0[:, :, ny // 2], vmin=-20, vmax=20, cmap='seismic')
    axs[1][0].set_title('Inverted')
    axs[1][0].axis('tight')
    axs[1][1].imshow(d[:, :, ny // 2] - u0[:, :, ny // 2],
                     vmin=-20, vmax=20, cmap='seismic')
    axs[1][1].set_title('Error')
    axs[1][1].axis('tight')
    fig.tight_layout()

    plt.savefig(data_output_path + "pylops_fft_3d_tv_recon.png")
    plt.close(fig)