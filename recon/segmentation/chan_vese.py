#
#
#
# Inspired by: Shawn Lankton (www.shawnlankton.com)
# github https://github.com/kevin-keraudren/chanvese

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from sklearn.metrics import mean_squared_error

from recon.helpers.image_boundary import boundary_handler_2d


# first of all - inputs are 2d images
eps = 0.0000000001
bh = boundary_handler_2d()

def bwdist(a):
    return nd.distance_transform_edt(a == 0)

# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + im2double(init_a) - 0.5
    return phi



def im2double(a):
    a = a.astype(np.float)
    a /= np.abs(a).max()
    return a

def chan_vese(img):
    """
    Description
    1. Initalize varphi_0 by phi_0, n = 0
    2. Compute c1(phi_n) and c2(phi_n) by (6) and (7)
    :return:
    """
    #n = 0
    #varphi_0 = 0
    #phi = 0
    #n, m = np.shape(img)
    return c1(img)

def c1(img):
    n, m = np.shape(img)

    phi = np.zeros((n,m))
    mask = np.zeros(img.shape)
    mask[5:n-5, 5:m-5] = 1

    phi = mask2phi(mask)
    prev_mask = phi
    its = 0
    max_its = 50
    stop = False


    while (its < max_its and not stop):
        # Get the curve's narrow band

        if True:

            if its % 10 == 0:
                fig, axes = plt.subplots(ncols=2)
                show_curve_and_phi(fig, img, phi, 'red')

            # Find interior and exterior mean
            inner = np.where(phi <= -eps)  # interior points
            outer = np.where(phi > eps)  # exterior points
            all = np.array([list(inner[0])+list(outer[0]), list(inner[1])+list(outer[1])]).T
            c1 = np.sum(img[inner]) / (sum(np.shape(inner)) + eps)  # interior mean
            c2 = np.sum(img[outer]) / (sum(np.shape(outer)) + eps)  # exterior mean

            a = img[inner]
            F = np.zeros((n,m))
            F[all[:, 0], all[:, 1]] = ((bh.get_value_at_idx(img, all) - c1) ** 2 -
                                       (bh.get_value_at_idx(img, all) - c2) ** 2).ravel()


            nu = 0
            dt = 0.5 #0.5
            mu = 20
            h = 1
            lambda1 = 1
            lambda2 = 1
            idx_list = []
            for i in range(n-1):
                for j in range(m-1):
                    idx_list.append((i, j))
            idx = np.array(idx_list)
            b= np.array([(1, 0)]*((n-1)*(m-1)))
            idx_xm1 = idx-np.array([(1, 0)]*((n-1)*(m-1)))
            idx_ym1 = idx-np.array([(0, 1)]*((n-1)*(m-1)))

            phi[idx[:, 0], idx[:, 1]] = (bh.get_value_at_idx(phi, idx)+dt*delta(bh.get_value_at_idx(phi, idx), h)*(


                    (coeff_A(phi, idx, mu, h) * (bh.dx_plus_at_idx(phi, idx)) -
                    coeff_A(phi, idx_xm1, mu, h) * (bh.dx_minus_at_idx(phi, idx_xm1))) +

                        (coeff_B(phi, idx, mu, h) * (bh.dy_plus_at_idx(phi, idx)) -
                        coeff_B(phi, idx_ym1, mu, h) * (bh.dy_minus_at_idx(phi, idx_ym1))) -
                        nu -
                        - bh.get_value_at_idx(F, idx)
                    )).ravel()


            its += 1

            new_mask = (phi <= 0)
            new_mask = new_mask.astype(int)

            mse = mean_squared_error(prev_mask, new_mask)
            if mse > eps:
                its = its + 1
                print("MSE: ", mse)
                prev_mask = new_mask
            else:
                stop = True

    return new_mask

def create_phi_0(x,y , n, m):
    a = (x-n/3)**2
    b = (y-m/3)**2
    return -np.sqrt(a+b)+min(n,m)/2

def h_curve(z, h=1, epsilon=0.001):
    if z > epsilon:
        return 1
    elif z <= -epsilon:
        return 0
    else:
        return 1/2*(1+z/epsilon+1/math.pi*np.sin(math.pi*z/epsilon))


def delta(z, epsilon=0.000000001, h = 1):
    #if abs(z) > epsilon:
    #    return 0
    #else:
    #return 1/(2*epsilon)*(1+z*np.cos(math.pi*z/epsilon))
    z[z > epsilon] = 1
    z[z < -epsilon] = 0
    z[(abs(z) >= 0) & (abs(z) <= epsilon)] = 1/(2*epsilon)*\
                                             (1+z[(abs(z) >= 0) & (abs(z) <= epsilon)]
                                                            *np.cos(math.pi*z[(abs(z) >= 0) & (abs(z) <= epsilon)]/epsilon))

    return z


def coeff_A(phi, idx, mu, h=1, eta=10e-8):
    return mu / (h**2 * np.sqrt(eta**2 + (bh.dx_plus_at_idx(phi, idx)/(h))**2 + (bh.dy_value_at_idx(phi, idx)/(2*h))**2))

def coeff_B(phi, idx, mu, h=1, eta=10e-8):
    return mu / (h**2 * np.sqrt(eta**2 + (bh.dy_plus_at_idx(phi, idx)/(h))**2 + (bh.dx_value_at_idx(phi, idx)/(2*h))**2))


# Displays the image with curve superimposed
def show_curve_and_phi(fig, I, phi, color):
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray')
    fig.axes[0].contour(phi, 0, colors=color)
    fig.axes[0].set_axis_off()
    plt.draw()

    fig.axes[1].cla()
    fig.axes[1].imshow(I, cmap='gray')
    fig.axes[1].set_axis_off()
    plt.draw()

    plt.pause(0.001)
