"""
07. White Noise Between Spaces
================

This Notebook examines the behavior of noise under linear operators numerically. Lets say for some $u,w \in H$, $K \in L(u,w)$ and $\eta$ some distributed noise:
\begin{equation}
    w = Ku \quad w_\eta = Ku + \eta
\end{equation}
Is the bias $\eta_u = (K^{-1}w- K^{-1}w_\eta)$ distributed?

"""

###############################################################################
# We import ....

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from recon.operator.ct_radon import CtRt

u = misc.face(gray=True)[256:256*3, 256:256*3]
u[:, 256:] = 0
u = u/np.max(u)
plt.imshow(u, cmap=plt.cm.gray)
plt.show()


###############################################################################
# 2

mu, sigma = 0, 0.1
eta_image = np.random.normal(mu, sigma, u.shape)
plt.imshow(u+eta_image, cmap=plt.cm.gray)
plt.show()
#%%
count, bins, ignored = plt.hist(eta_image.ravel(), 50, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

###############################################################################
# 3
theta = list(np.linspace(0., 180., 25, endpoint=False))

R = CtRt(np.shape(u),
         np.array([(np.shape(u)[0]/2)+1, (np.shape(u)[0]/2)+1]),
         theta=theta)
w = R*u.ravel()
w0 = R*(u+eta_image).ravel()
eta_w = w-w0

count, bins, ignored = plt.hist(eta_w.ravel(), 50, density=True)
plt.show()

###############################################################################
## Backward

mu, sigma = 0, 0.01*np.max(w)
eta = np.random.normal(mu, sigma, w.shape)
w_eta = w + eta
plt.imshow(np.reshape(R.inv*w_eta.ravel(), u.shape), cmap=plt.cm.gray)
plt.show()
#%%
recon_eta_est = (np.reshape(R.inv*w_eta.ravel(), u.shape)-u).ravel()
count, bins, ignored = plt.hist(recon_eta_est, 50, density=True)
plt.show()
#%%
print("Backwards-Image-Mean: "+str(round(np.mean(recon_eta_est), 4)))
print("Backwards-Image-Sigma: "+str(round(np.std(recon_eta_est, ddof=1), 4)))

###############################################################################
## Only Noise

mu, sigma = 0, 0.01*np.max(w)
eta = np.random.normal(mu, sigma, w.shape)
eta_est = R.inv*eta.ravel()
sigma_est = np.std(eta_est, ddof=1)
mu_est = np.mean(eta_est)
count, bins, ignored = plt.hist(eta_est, 50, density=True)
plt.plot(bins, 1/(sigma_est * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu_est)**2 / (2 * sigma_est**2)),
         linewidth=2, color='r')
plt.show()
print("Only-Noise-Recon:Mean: "+str(round(np.mean(eta_est), 4)))
print("Only-Noise-Recon-Sigma: "+str(round(np.std(eta_est, ddof=1), 4)))

###############################################################################
## Result
print("Mean Difference: "+str(np.abs(np.mean(eta_est) - np.mean(recon_eta_est))))
print("Sigma Difference: "+str(np.abs(np.std(eta_est, ddof=1) - np.std(recon_eta_est, ddof=1))))

###############################################################################
#
# Therefore for a normal distributed $\eta \sim \mathcal N(0, \sigma_0)$:
# \begin{equation}
#     K^{-1}(w+\eta)-K^{-1}(w) \sim K^{-1}\eta \sim \mathcal N(0, \sigma_1)
# \end{equation}
#
# \sigma_1 has to be calculated.

###############################################################################
## Shape independent
mu, sigma = 0, 0.01*np.max(w)
for shape in [(128, 128), (256, 256), (512, 512)]:
    R = CtRt(shape,
             np.array([(shape[0] // 2), (shape[0] // 2)]),
             theta=theta)
    eta = np.random.normal(mu, sigma, R.image_dim)
    eta_est = R.inv*eta.ravel()
    print(str(shape) + "-Mean: "+str(round(np.mean(eta_est), 4)))
    print(str(shape) +"-Sigma: "+str(round(np.std(eta_est, ddof=1),4)))

###############################################################################
## Estimation of the background
w = R*u.ravel()
w0 = R*(u+eta_image).ravel()
background = np.reshape((R.inv * w0), u.shape) [:, int(256+1):]
sigma_est = np.mean(background**2)
mu_est = np.mean(background)
print("BACKGROUND-Sigma: "+str(sigma_est))
print("BACKGROUND-MU: "+str(mu_est))
