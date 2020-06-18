import numpy as np
from scipy import sparse

from recon.terms import Projection
from recon.terms.dataterm_linear import DatatermLinear
from recon.solver.pd_hgm import PdHgm
from experimental.helpers import normest


def multi_class_segmentation_bregman(img,
                                     classes: list,
                                     beta: float= 0.001,
                                     delta: float = 1,
                                     qk = None,
                                     tau: float = None):

    f = np.zeros(((img.shape[0], img.shape[1], len(classes))))
    raveld_f =  np.zeros(((img.shape[0]*img.shape[1], len(classes))))

    for i in range(len(classes)):
        #f[:, :, i] = (img.T - classes[i]) ** 2
        raveld_f[:,i] = delta * (img.ravel() - classes[i]) ** 2 - beta * (qk[:, i])

    #f = np.ravel(f, order='C')
    f = raveld_f

    shape = (img.shape[0], img.shape[1])

    dx = sparse.diags([1, -1], [0, 1], shape=(shape[1], shape[1])).tocsr()
    dx[shape[1] - 1, :] = 0
    dy = sparse.diags([-1, 1], [0, 1], shape=(shape[0], shape[0])).tocsr()
    dy[shape[0] - 1, :] = 0

    grad = sparse.vstack((sparse.kron(dx, sparse.eye(img.shape[0]).tocsr()),
                          sparse.kron(sparse.eye(img.shape[1]).tocsr(), dy)))

    boundaries = 'neumann'
    # grad = FirstDerivative(262144, boundaries=boundaries)
    K = beta * grad
    # vd1 = convex_segmentation(u0, beta0, classes)

    G = DatatermLinear()
    G.set_proxdata(f)
    F_star = Projection(f.shape)
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
    solver.maxiter = 150


    solver.tol = 10 ** (-6)

    solver.solve()

    seg = np.reshape(solver.var['x'], (img.shape[0], img.shape[1], len(classes)), order='C')

    # set(figure,'defaulttextinterpreter','latex');
    a = seg
    result = np.zeros(a.shape)
    for i in range(a.shape[0]):  # SLOW
        for j in range(a.shape[1]):
            idx = np.argmin((a[i, j, :]))
            result[i, j, idx] = 1

    result0 = sum([i*result[:, :, i] for i in range(len(classes))])


    return result0, result
