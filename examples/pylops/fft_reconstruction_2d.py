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

from recon.math.terms import Dataterm, Projection
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest

import pylops

plt.close('all')

data_output_path = "./../data/pylops/output/"

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
tv_recon = True
recalc_normest = True
if tv_recon:
    ny = 100 #2000
    nx = 20  #20480
    #grad = pylops.FirstDerivative(nt * nx, dims=(nt, nx), dir=0, dtype='float64')
    grad = pylops.Gradient(dims=(nx, ny), dtype='float64')
    alpha0=2.0
    K = alpha0*grad

    if recalc_normest:
        norm = normest(K)
        sigma0=0.99/norm
        print(sigma0)
    else:
        sigma0= 3.522
    tau0 = sigma0

    G = Dataterm(S=None, F=FFTop)
    F_star = Projection(d.shape)
    solver = PdHgm(K, F_star, G)
    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 1500
    solver.tol = 5*10**(-12)

    G.set_proxdata(D)
    solver.solve()

    u0 = np.reshape(np.real(solver.var['x']), d.shape)
    rel_tvrec = np.linalg.norm(d - u0, 2) #/np.linalg.norm(gt)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0][0].imshow(d, vmin=-20, vmax=20, cmap='seismic')
    axs[0][0].set_title('Signal')
    axs[0][0].axis('tight')
    axs[0][1].imshow(np.abs(D.reshape(nfft, nx)[:200, :]), cmap='seismic')
    axs[0][1].set_title('Fourier Transform')
    axs[0][1].axis('tight')
    axs[1][0].imshow(u0, vmin=-20, vmax=20, cmap='seismic')
    axs[1][0].set_title('Inverted')
    axs[1][0].axis('tight')
    axs[1][1].imshow(d - u0, vmin=-20, vmax=20, cmap='seismic')
    axs[1][1].set_title('Error')
    axs[1][1].axis('tight')
    fig.tight_layout()

    plt.savefig(data_output_path + "pylops_fft_tv_recon.png")
    plt.close(fig)