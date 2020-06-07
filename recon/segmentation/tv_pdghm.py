import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pylops


from recon.math.terms import Projection, DatatermLinear
from recon.math.operator.first_derivative import FirstDerivative
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest


def multi_class_segmentation(img, classes: list, beta: float= 0.001, tau: float= None):

    #f = np.zeros( tuple( list(img.shape) + [len(classes)]))
    raveld_f =  np.zeros(((np.prod(img.shape), len(classes))))

    for i in range(len(classes)):
        #f[:, :, i] = (img.T - classes[i]) ** 2
        raveld_f[:,i] = (img.ravel() - classes[i]) ** 2

    #f = np.ravel(f, order='C')
    f = raveld_f


    grad = pylops.Gradient(dims=img.shape, dtype='float64')
    # grad = FirstDerivative(262144, boundaries=boundaries)
    K = beta * grad
    # vd1 = convex_segmentation(u0, beta0, classes)

    G = DatatermLinear()
    G.set_proxdata(f)
    F_star = Projection(f.shape, len(img.shape))
    solver = PdHgm(K, F_star, G)

    solver.var['x'] = np.zeros((K.shape[1], len(classes)))
    solver.var['y'] = np.zeros((K.shape[0], len(classes)))

    if tau:
        tau0 = tau
    else:
        tau0 = 0.99 / normest(K)
        print(tau0)
    sigma0 = tau0

    G.set_proxparam(tau0)
    F_star.set_proxparam(sigma0)
    solver.maxiter = 200
    solver.tol = 10 ** (-6)

    # G.set_proxdata(f)
    solver.solve()

    seg = np.reshape(solver.var['x'], tuple( list(img.shape) + [len(classes)]), order='C')

    a = seg
    result = [] #np.zeros(a.shape)
    #for i in range(a.shape[0]):  # SLOW
    #    for j in range(a.shape[1]):
    #            idx = np.argmin((a[i, j,  :]))
    #            result[i, j, idx] = 1
    tmp_result = np.argmin(a, axis=len(img.shape))

    for i,c in enumerate(classes):
        result.append((tmp_result == i).astype(int))

    result0 = sum([i*result[i] for i in range(len(classes))])

    result = np.array(result)

    return result0, result
