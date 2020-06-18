"""
################
# EXPERIMENTAL #
################

First try of some new thoughts:
Local Regularization
In some areas more TV than in others.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


from recon.reconstruction import PdSmooth
from recon.reconstruction.pd_smooth_mtv import PdSmoothMTV
from pylops import FirstDerivative, Smoothing2D, Gradient, Laplacian

data_import_path = "./data/"
data_output_path = data_import_path+"output/"

img = misc.ascent()
img = img/np.max(img)
gt = img[150:480, 120:450]
sigma = 0.15
n = sigma*np.max(abs(gt.ravel()))*np.random.normal(size=gt.shape)
#n = n - (n<0).astype(int) * 0.02 + (n>=0).astype(int)*0.02
#n[:-50,:-50] = 0
noise_img = gt + n

def draw_images(a, name, error=None ,max=np.max(gt)):
        plt.gray()
        plt.imshow(a, vmin=np.min(gt), vmax=max)
        plt.axis('off')
        if error !=  None:
                plt.title("NE: {:2f}".format(error))
        plt.savefig(data_output_path + name, bbox_inches='tight', pad_inches=0)
        plt.close()

draw_images(noise_img, "noise.png", np.linalg.norm(gt - noise_img, 1))




# only right sided TV smoothing for principle
alpha = 0.13
#alpha[:,:gt.shape[1]//2] = 0
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha)
#u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
#draw_images(u0, "normal_tv_regularization.png", np.linalg.norm(gt - u0, 1))

"""
alpha = np.ones(gt.shape)*0.05
tv_smoothing = PdSmoothMTV(domain_shape=gt.shape, reg_mode='tv', alpha=alpha, data_output_path=data_output_path, noise_sigma=sigma)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "normal_tv_regularization_mtv.png", np.linalg.norm(gt - u0, 1))
"""





lap = Laplacian(gt.shape)
grad = Gradient(gt.shape, edge=True, kind='backward')
fd0 = FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir = 0, kind="backward")
fd1 = FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir = 1, kind="backward")

#test = np.reshape(lap * gt.ravel(), gt.shape)
test = grad * gt.ravel()
#test[test > 0]= 1
#test[test < 0]= -1
#test[test == 0]= 0

#test[grad * gt.ravel() < 0.00001] = 0



#test[np.isnan(test)] = 10000

test0 = fd0 * test[:np.prod(gt.shape)] + fd1 * test[np.prod(gt.shape):]
#test0 = fd0 * np.abs(test[:np.prod(gt.shape)]) + fd1 * np.abs(test[np.prod(gt.shape):])

#test0 = (lap * gt.ravel()) * 5 #/ np.sum(np.abs(grad * gt.ravel()))

test0 = np.clip(test0, 0.01, 1000000000)
test0 *= 50


draw_images(np.reshape(test0, gt.shape), "fun2.png")

#test0[np.isnan(test0)] = 0


#test0 = np.clip(test0, 0, 1)

draw_images(np.reshape(test0, gt.shape), "gt2.png")

#lam = np.abs(gt - noise_img) / np.reshape(test0, gt.shape)
s = 1
lam = (s*np.abs(gt - noise_img)) / np.reshape(test0, gt.shape)

#lam = lam[:70, :5]

#lam[np.abs(np.reshape(test0, gt.shape))<0.000001] = 0.5

from numpy import inf

#lam[np.isnan(lam)] = 10000
#lam[lam == -inf] = 0
#lam[np.abs(gt - noise_img)<0.000001] = 0
lam = np.abs(lam)
#lam[lam == inf] = 5

#lam = lam/np.max(lam) * 0.8

#lam = np.clip(lam, 0, 1)


Sop = Smoothing2D(nsmooth=[5, 5], dims=gt.shape, dtype='float64')
#lam = 0.01 + 0.5* np.abs(np.reshape(fd0*noise_img.ravel(), gt.shape)) + 0.5*np.abs(np.reshape(fd1*noise_img.ravel(), gt.shape))
lam = np.ones(gt.shape)*0.3 \
        - 1 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=0)*noise_img.ravel(), gt.shape)) \
        - 1 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=1)*noise_img.ravel(), gt.shape))
#alpha = 1 - gt/np.max(gt)

#lam = lam/np.mean(lam)*0.2

draw_images(lam, "fun.png")
draw_images(gt, "gt.png", error=0)
#draw_images(noise_img[:70, :5], "noise.png")
#draw_images(np.reshape(test0, gt.shape)[:70, :5], "gt2.png")
#draw_images(np.abs(gt - noise_img)[:70, :5], "gt3.png")




tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=lam)
u0 = tv_smoothing.solve(data=noise_img, maxiter=250, tol=5*10**(-4))
draw_images(u0, "normal_tv_regularization_a.png", np.linalg.norm(gt - u0, 1))

draw_images(np.abs(u0-gt), "normal_tv_regularization_error")

print("test")



"""
# only right sided TV smoothing for principle
alpha = np.ones(gt.shape)*0.5
alpha[:,:gt.shape[1]//2] = 0
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha, tau=0.875)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "splitted_regularization.png", np.linalg.norm(gt - u0, 2))


# Gradient based local regularization - decreasing Gradient influence
Sop = Smoothing2D(nsmooth=[5, 5], dims=gt.shape, dtype='float64')

alpha = np.ones(gt.shape)*1 \
        - 1.2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=0)*noise_img.ravel(), gt.shape)) \
        - 1.2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=1)*noise_img.ravel(), gt.shape))
#alpha = 1 - gt/np.max(gt)
#alpha = np.clip(alpha,0.2, 0.5)
alpha = alpha/np.mean(alpha)*0.5
alpha = np.clip(alpha,0.3, 0.6)
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha, tau=0.8335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "splitted_regularization_1.png", np.linalg.norm(u0-gt, 2))
draw_images(alpha, "splitted_regularization_1alpha.png")
"""
"""
# only left sided TV smoothing - increasing Gradient influence
Sop = Smoothing2D(nsmooth=[3, 3], dims=gt.shape, dtype='float64')

alpha = np.ones(gt.shape)*0.4 \
        + 2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=0)*gt.ravel(), gt.shape)) \
        + 2 * np.abs(np.reshape(Sop*FirstDerivative(np.prod(gt.shape), dims=gt.shape, dir=1)*gt.ravel(), gt.shape))
alpha = alpha/np.mean(alpha)*0.6
tv_smoothing = PdSmooth(domain_shape=gt.shape, reg_mode='tv', alpha=alpha) #, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
draw_images(u0, "splitted_regularization_2.png")
draw_images(alpha, "splitted_regularization_2alpha.png")
"""
"""
# some not implemented alternative algorithms...
alpha = np.ones(gt.shape)*0.1
tv_smoothing = PdSmoothSPTV(domain_shape=gt.shape,
                            reg_mode='tv',
                            alpha=alpha,
                            noise_sigma=np.sqrt(sigma),
                            data_output_path=data_output_path) #, tau=2.3335)
u0 = tv_smoothing.solve(data=noise_img, maxiter=150, tol=5*10**(-4))
plt.gray()
plt.imshow(u0, vmin=0, vmax=np.max(gt))
plt.axis('off')
plt.savefig(data_output_path+'2d_local_smoothing_tv.png', bbox_inches = 'tight', pad_inches = 0)
plt.close()
"""