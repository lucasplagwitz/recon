import numpy as np
import matplotlib.pyplot as plt
import imageio

from pylops import Gradient

from recon.interfaces import Recon
from recon.operator import CtRt

# load content
image_path = "./../../examples/data/gt.png"
gt = imageio.imread(image_path, as_gray=True)
gt = gt/np.max(gt)
gt = gt[:75, :75]
#gt[:,:] = 0
gt[20:40, 20:60] = 1


theta = list(np.linspace(0., 180., 360, endpoint=False))

F = CtRt(np.shape(gt),
         np.array([(np.shape(gt)[0]/2)+1, (np.shape(gt)[0]/2)+1]),
         theta=theta)


g = F * gt.ravel()

# Gaussian noise
sigma = 0
n = sigma*np.max(np.abs(g))*np.random.normal(size=g.shape)
f = g + n

grad = Gradient(gt.shape, edge=True, dtype='float64', kind='backward')

plt.imshow(gt)
plt.axis('off')
plt.savefig('./data/gt.png', bbox_inches = 'tight', pad_inches = 0)


#
data_output_path = "./data/output2/"
#alpha = np.ones((107,180))*0.4
alpha = np.ones((75, 75))*0.01
alpha[:,alpha.shape[1]//2:] = 1
#alpha = 0.1
"""
tv_smoothing = Recon(F,
                     domain_shape=gt.shape,
                     #image_shape=(107,180),
                     reg_mode='tv', alpha=alpha,
                     weight_term='reg')
u0 = tv_smoothing.solve(data=f, max_iter=150, tol=5*10**(-4))
plt.imshow(np.reshape(u0, gt.shape))
plt.axis('off')
plt.colorbar()
plt.savefig('./data/tv_recon_old.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(alpha)
plt.axis('off')
plt.colorbar()
plt.savefig('./data/alpha_reg.png', bbox_inches = 'tight', pad_inches = 0)
"""
#alpha = 10.0
#alpha = np.ones((107, 180))*10
alpha = np.ones((75, 75))*0
alpha[:,:alpha.shape[1]//2] = 1
alpha = np.reshape(F*((alpha).ravel()), F.image_dim)
alpha = alpha/np.max(alpha)
#alpha[alpha < 0.99]  = 1
alpha[alpha > 0.9] = 0

plt.imshow(alpha)
plt.axis('off')
plt.colorbar()
plt.savefig('./data/alpha_1.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.reshape(F.H*alpha.ravel(), (75,75)))
plt.axis('off')
plt.colorbar()
plt.savefig('./data/alpha_2.png', bbox_inches = 'tight', pad_inches = 0)

tv_smoothing = Recon(F,
                     domain_shape=gt.shape,
                     #image_shape=(107,180),
                     tau = 0.35,
                     reg_mode='tv', alpha=alpha,
                     weight_term='data')
u0 = tv_smoothing.solve(data=f, max_iter=150, tol=5*10**(-4))



plt.imshow(alpha)
plt.axis('off')
plt.colorbar()
plt.savefig('./data/alpha_data.png', bbox_inches = 'tight', pad_inches = 0)

plt.imshow(np.reshape(u0, gt.shape))
plt.axis('off')
plt.colorbar()
plt.savefig('./data/tv_recon_new.png', bbox_inches = 'tight', pad_inches = 0)
