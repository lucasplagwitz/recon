# Based on exmaple from: https://github.com/equinor/pylops
import numpy as np
import matplotlib.pyplot as plt
import pylops

from recon.interfaces import Recon

plt.close('all')

data_output_path = "./data/output/"

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
tv_recon = Recon(O=FFTop,
                 domain_shape=d.shape,
                 reg_mode='tv',
                 alpha=15.0)

u = np.real(tv_recon.solve(D))

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d[:, :, ny//2], vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(np.fft.fftshift(D.reshape(nfft, nfftk, nfftk),
                                        axes=1)[:20, :, nfftk // 2]),
                     cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(u[:, :, ny // 2], vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d[:, :, ny // 2] - u[:, :, ny // 2],
                     vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path + "pylops_fft_3d_tv_recon.png")
plt.close(fig)