"""
Fourier Transform
=================
This example shows how to use the :py:class:`pylops.signalprocessing.FFT`,
:py:class:`pylops.signalprocessing.FFT2D`
and :py:class:`pylops.signalprocessing.FFTND` operators to apply the Fourier
Transform to the model and the inverse Fourier Transform to the data.
"""
import numpy as np
import matplotlib.pyplot as plt

from recon.reconstruction import PdRecon

import pylops

plt.close('all')

data_output_path = "./data/pylops/output/"

dt = 0.005
nt, nx = 100, 20
t = np.arange(nt)*dt
f0 = 10
nfft = 2**10
d = np.outer(np.sin(2*np.pi*f0*t), np.arange(nx)+1)

FFTop = pylops.signalprocessing.FFT(dims=(nt, nx), dir=0,
                                    nfft=nfft, sampling=dt)

# gt exmaple of pylops #
#######################
D = FFTop*d.flatten()

# Adjoint = inverse for FFT
dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(nt, nx)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d, vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(D.reshape(nfft, nx)[:200, :]), cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv, vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d-dinv, vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path+"pylops_fft_plain.png")
plt.close(fig)



# Gaussian noise #
##################
D = FFTop*d.flatten()
sigma = 0.1
n = sigma*np.max(np.abs(D))*np.random.normal(size=D.shape[0])
D = D + n

# Adjoint = inverse for FFT
dinv = FFTop.H*D
dinv = FFTop / D
dinv = np.real(dinv).reshape(nt, nx)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d, vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(D.reshape(nfft, nx)[:200, :]), cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(dinv, vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d-dinv, vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path+"pylops_fft_noise.png")
plt.close(fig)


# Gaussian noise - TV regularised Reconstruction #
##################################################
tv_recon = PdRecon(O=FFTop,
                   domain_shape=d.shape,
                   reg_mode='tv',
                   alpha=2.0,
                   tau=None)


u = tv_recon.solve(D)
rel_tvrec = np.linalg.norm(d - u, 2) #/np.linalg.norm(gt)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0][0].imshow(d, vmin=-20, vmax=20, cmap='seismic')
axs[0][0].set_title('Signal')
axs[0][0].axis('tight')
axs[0][1].imshow(np.abs(D.reshape(nfft, nx)[:200, :]), cmap='seismic')
axs[0][1].set_title('Fourier Transform')
axs[0][1].axis('tight')
axs[1][0].imshow(u, vmin=-20, vmax=20, cmap='seismic')
axs[1][0].set_title('Inverted')
axs[1][0].axis('tight')
axs[1][1].imshow(d - u, vmin=-20, vmax=20, cmap='seismic')
axs[1][1].set_title('Error')
axs[1][1].axis('tight')
fig.tight_layout()

plt.savefig(data_output_path + "pylops_fft_tv_recon.png")
plt.close(fig)