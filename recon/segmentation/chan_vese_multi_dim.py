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
import itertools
import copy

from recon.helpers.image_boundary import boundary_handler_2d


# first of all - inputs are 2d images
eps = 1
bh = boundary_handler_2d()

def bwdist(a):
    return nd.distance_transform_edt(a == 0)

# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + im2double(init_a) - 0.5
    return phi

def roundphi(n = 256,m = 256, every=5):
    helper = np.zeros((n, m))
    helper[every::30, every::15] = 1
    phi = bwdist(helper) - bwdist(1 - helper) + helper - 5
    phi = phi/np.max(phi)*4
    return phi


def im2double(a):
    a = a.astype(np.float)
    a /= np.abs(a).max()
    return a

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def calc_average(c_matrix):
    """

    :param a:
    :return:
    """


def chan_vese(img, n_segments=3, classes=[0, 200, 80, 200], nu=0.0001):
    n, m = np.shape(img)

    n_phis = n_segments - 1

    phi = np.zeros((n, m, n_phis))

    # compute (n-1) start phi for n segments
    diff = 20
    for seg in range(n_phis):
        mask = np.zeros(img.shape)
        mask[diff*(seg + 1):n-diff*(seg + 1), diff*(seg + 1):m-diff*(seg + 1)] = 1
        #phi[:,:,seg] = mask2phi(mask)
        phi[:, :, seg] = roundphi(256, 256, every=5*seg+5)

    c_matrix = np.zeros((2**(n_phis)))

    prev_mask = phi
    its = 0
    max_its = 40
    stop = False


    while (its < max_its and not stop):
        # Get the curve's narrow band

        if True:

            if its % 2 == 0:
                fig, axes = plt.subplots(ncols=2, nrows=2)
                show_curve_and_phi(fig, img, phi, 'red')

            # Find interior and exterior mean
            #inner = [0]*n_phis
            #outer = [0]*n_phis
            #outer = [0]*n_phis
            #for seg in range(n_phis):
            all_all = np.array(np.where((phi < 0) | (phi >= 0)))

            # first approach
            """
            inner = np.where(phi > 1)  # interior points
            inner = np.vstack([inner, np.ones(np.shape(inner)[1])])
            outer = np.where(phi < -1)  # exterior points
            outer = np.vstack([outer, np.zeros(np.shape(outer)[1])])

            all = np.concatenate([inner, outer], axis=1)

            combination = [list(i) for i in itertools.product([0, 1], repeat=n_phis)]


            index = {}
            lambda_functions = {}
            cost = {}
            for com in combination:
                com_name = str(com)
                index[com_name] = all[:2, :].T
                cond = True
                for i, c in enumerate(com):
                    tmp_index = all[:, ((all[2, :] == np.ones(np.shape(all)[1])*i) &
                                                           ((all[3, :] == np.ones(np.shape(all)[1])*c)))]
                    index[com_name] = set([tuple(x) for x in list(index[com_name])]).intersection(
                        set([tuple(x) for x in list(tmp_index[:2,:].T)]))
                    #                                 ]

                    #index[com_name] = np.concatenate([index[com_name],
                    #                                 all[:, ((all[2, :] == np.ones(np.shape(all)[1])*i) &
                    #                                         ((all[3, :] == np.ones(np.shape(all)[1])*c)))]
                    #                                 ], axis=1)
                    #index[com_name].append()
                #index[com_name] = unique_rows(index[com_name][:2, :].T).astype(int).T
                #index[com_name] = np.array([list(index[com_name][0]), list(index[com_name][1])]).T
                index[com_name] = np.array(list(index[com_name])).astype(int)
                if len(index[com_name]) > 0:
                    cost[com_name] = np.sum(bh.get_value_at_idx(img, index[com_name])) / (np.shape(index[com_name])[0])
                else:
                    print("fail")
                    cost[com_name] = 0

            print(cost)
            if classes:
                for i,j in enumerate(cost.keys()):
                    if classes[i] is not None:
                        cost[j] = classes[i]
            print(cost)
            
            
            combination = [list(i) for i in itertools.product([0, 1], repeat=n_phis)]
            index = {}
            cost = {}
            for com in combination:
                com_name = str(com)
                index[com_name] = all[:2, :].T
                cond = True
                for i, c in enumerate(com):
                    tmp_index = all[:, ((all[2, :] == np.ones(np.shape(all)[1]) * i) &
                                        ((all[3, :] == np.ones(np.shape(all)[1]) * c)))]
                    index[com_name] = set([tuple(x) for x in list(index[com_name])]).intersection(
                        set([tuple(x) for x in list(tmp_index[:2, :].T)]))
                    #                                 ]

                    # index[com_name] = np.concatenate([index[com_name],
                    #                                 all[:, ((all[2, :] == np.ones(np.shape(all)[1])*i) &
                    #                                         ((all[3, :] == np.ones(np.shape(all)[1])*c)))]
                    #                                 ], axis=1)
                    # index[com_name].append()
                # index[com_name] = unique_rows(index[com_name][:2, :].T).astype(int).T
                # index[com_name] = np.array([list(index[com_name][0]), list(index[com_name][1])]).T
                index[com_name] = np.array(list(index[com_name])).astype(int)
                if len(index[com_name]) > 0:
                    cost[com_name] = np.sum(bh.get_value_at_idx(img, index[com_name])) / (np.shape(index[com_name])[0])
                else:
                    print("fail")
                    cost[com_name] = 0
            """

            nu = 0.000001 * np.max(img)**2
            dt = 0.1 #0.5
            mu = 12
            h = 1
            lambda1 = 1
            lambda2 = 1
            cost = {}

            for k_phis in range(n_phis):

                #a[0] = (1-h_curve())
                #a[1] =

                one_m_h_term = np.zeros(img.shape)
                h_term = np.zeros(img.shape)
                F = np.zeros((n, m, n_phis))
                #all_ex = all[:, ((all[2, :] == np.ones(np.shape(all)[1]) * k_phis))]
                all_ex = np.unique(all_all[:2, :], axis=1).astype(int)
                all_ex = np.array([list(all_ex[0]), list(all_ex[1])]).T.astype(int)

                #all[:, ((all[2, :] == np.ones(np.shape(all)[1])*i) &
                #         ((all[3, :] == np.ones(np.shape(all)[1])*c)))]
                combination = [list(i) for i in itertools.product([0, 1], repeat=n_phis)]

                for com in combination:
                    com_name = str(com)
                    prod_value = 1
                    for k in range(n_phis):
                        if com[k] == 1:
                            prod_value *= h_curve(phi[:, :, k])
                        else:
                            prod_value *= (1 - h_curve(phi[:, :, k]))

                    cost[com_name] = np.sum(img * prod_value) / np.sum(prod_value)

                print(cost)
                if classes:
                    for i, j in enumerate(cost.keys()):
                        if classes[i] is not None:
                            cost[j] = abs(classes[i]-cost[j])
                print(cost)

                for com in combination:
                    com_name = str(com)
                    # for c in com:
                    if com[k_phis] == 1:
                        sgn = -1
                    else:
                        sgn = 1

                    if com[(k_phis+1)%n_phis] == 0:
                        one_m_h_term += sgn * (img - cost[com_name]) ** 2
                    else:
                        h_term += sgn * (img - cost[com_name]) ** 2


                one_m_h_term = one_m_h_term * (1 - h_curve(phi[:, :, (k_phis+1)%n_phis]))
                h_term = h_term * h_curve(phi[:, :, (k_phis+1)%n_phis])

                #F[all[:, 0], all[:, 1], k_phis] = (bh.get_value_at_idx(F, all) +
                #                                      (bh.get_value_at_idx(img, all) - cost[com_name]) ** 2
                #                                      * bh.get_value_at_idx(base, all) ).ravel()
                    #F = old_F + F
                #F[all_ex[:, 0], all_ex[:, 1], k_phis] = (bh.get_value_at_idx(h_term, all_ex) +
                #                                         bh.get_value_at_idx(one_m_h_term, all_ex)).ravel()

                F[:, :, k_phis] = h_term + one_m_h_term

            #for k_phis in range(n_phis):
                idx_list = []
                for i in range(n):
                    for j in range(m):
                        idx_list.append((i, j))
                idx = np.array(idx_list)
                idx_xm1 = idx - np.array([(1, 0)]*((n)*(m)))
                idx_xp1 = idx + np.array([(1, 0)] * ((n) * (m)))
                idx_ym1 = idx - np.array([(0, 1)]*((n)*(m)))
                idx_yp1 = idx + np.array([(0, 1)] * ((n) * (m)))

                m1 = nu * dt*delta(bh.get_value_at_idx(phi[:, :, k_phis], idx), h)
                C1 = coeff_A(phi[:, :, k_phis], idx, mu, h)
                C2 = coeff_A(phi[:, :, k_phis], idx_xm1, mu, h)
                C3 = coeff_B(phi[:, :, k_phis], idx, mu, h)
                C4 = coeff_B(phi[:, :, k_phis], idx_ym1, mu, h)
                C = 1+ m1 * (C1 + C2 + C3 + C4)

                phi[idx[:, 0], idx[:, 1], k_phis] = (1/C * (bh.get_value_at_idx(phi[:, :, k_phis], idx) +
                                                            m1 * (
                                                             C1 * (bh.get_value_at_idx(phi[:, :, k_phis], idx_xp1)) +
                                                             C2 * (bh.get_value_at_idx(phi[:, :, k_phis], idx_xm1)) +
                                                             C3 * (bh.get_value_at_idx(phi[:, :, k_phis], idx_yp1)) +
                                                             C4 * (bh.get_value_at_idx(phi[:, :, k_phis], idx_ym1))
                                                            ) + dt*delta(bh.get_value_at_idx(phi[:, :, k_phis], idx), h)
                                                            * bh.get_value_at_idx(F[:,:,k_phis], idx)
                                                            )
                                                     ).ravel()


            its += 1

            new_mask = (phi <= 0)
            new_mask = new_mask.astype(int)

            mse = 0
            for k_phis in range(n_phis):
                mse += mean_squared_error(prev_mask[:,:,k_phis], new_mask[:,:,k_phis])
            if mse > 0.001:
                its = its + 1
                print("MSE: ", mse)
                prev_mask = new_mask
            else:
                stop = True
                #pass

    result0 = sum([(i+1) * new_mask[:, :, i] for i in range(n_phis)])

    return result0




def create_phi_0(x,y , n, m):
    a = (x-n/3)**2
    b = (y-m/3)**2
    return -np.sqrt(a+b)+min(n,m)/2


def h_curve(z, h=1, epsilon=1):
    #epsilon = eps
    zz = copy.copy(z)
    #zz[zz > epsilon] = 1
    #zz[zz < -epsilon] = 0
    zz = 1 / (2) *\
                                             (1+2/math.pi * np.arctan(zz / epsilon))
    return zz



def delta(z, epsilon=1, h = 1):
    #epsilon = eps
    #if abs(z) > epsilon:
    #    return 0
    #else:
    #return 1/(2*epsilon)*(1+z*np.cos(math.pi*z/epsilon))
    zz = copy.copy(z)
    zz = 1/(math.pi) * \
                                             (epsilon / (epsilon**2 + zz ** 2))
    # [(abs(zz) > 0) & (abs(zz) <= epsilon)]
    return zz


def coeff_A(phi, idx, mu, h=1, eta=10e-8):
    return mu / (h**2 * np.sqrt(eta**2 + (bh.dx_plus_at_idx(phi, idx)/(h))**2 + (bh.dy_value_at_idx(phi, idx)/(2*h))**2))

def coeff_B(phi, idx, mu, h=1, eta=10e-8):
    return mu / (h**2 * np.sqrt(eta**2 + (bh.dy_plus_at_idx(phi, idx)/(h))**2 + (bh.dx_value_at_idx(phi, idx)/(2*h))**2))


# Displays the image with curve superimposed
def show_curve_and_phi(fig, I, phi, color):
    eps = 0
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray', vmin=0, vmax=np.max(I))
    _, _, n_seg = np.shape(phi)
    colors = ['red', 'green', 'blue', 'yellow']
    for seg in range(n_seg):
        fig.axes[0].contour(phi[:,:, seg], 0, colors=colors[seg])
    fig.axes[0].set_axis_off()
    plt.draw()

    # phi0 > 0 phi1 < 0
    copyI = np.zeros(np.shape(I))
    cond = ((phi[:, :, 0] > eps))
    copyI[cond] = I[cond]

    fig.axes[1].cla()
    fig.axes[1].imshow(copyI, cmap='gray', vmin=0, vmax=np.max(I))
    fig.axes[1].set_axis_off()
    plt.draw()

    # phi0 < 0 phi1 > 0
    copyI = np.zeros(np.shape(I))
    cond = ((phi[:, :, 1] > eps))
    copyI[cond] = I[cond]

    fig.axes[2].cla()
    fig.axes[2].imshow(copyI, cmap='gray', vmin=0, vmax=np.max(I))
    fig.axes[2].set_axis_off()
    plt.draw()

    # phi0 > 0 phi1 < 0
    copyI = np.zeros(np.shape(I))
    cond = ((phi[:, :, 1] < -eps))
    copyI[cond] = I[cond]

    fig.axes[3].cla()
    fig.axes[3].imshow(copyI, cmap='gray', vmin=0, vmax=np.max(I))
    fig.axes[3].set_axis_off()
    plt.draw()

    plt.pause(0.001)
