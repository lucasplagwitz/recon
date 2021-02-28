"""
Primal Dual Algorithm based on Paper:
A first-order primal-dual algorithm for convex problems with applications to imaging
by Antonin Chambolle, Thomas Pock.

Implementation is based on Veronica Corona et al.
"""
import numpy as np
import copy
import sys
import pylops
from typing import Union
import progressbar
import time

from recon.terms import BaseDataterm, BaseRegTerm, IndicatorL2
from recon.solver import PdHgm


class PdHgmExtend(object):
    """
    Primal Dual Solver. Special form for PdHgm

    argmin G(Ax) + F(Kx) 

    Algorithm: (See /documentation/primal_dual.pdf for derivation)
            while tol < sens and max_iter not reached
                1. x^{n+1} = prox_{G}( x^{n} - phi * (K.T * y^{n}) )
                2. y^{n+1} = prox_{F_star}( y^{n} - sigma * (K * (2*x^{n+1} - x^{n})) )
                3. update sens

    Parameter
    ---------
    G: BaseDataterm

    F_star: BaseRegTerm

    K: pylops.LineareOperator

    gamma: float default: None
        stepsize for adaption rule for tau and sigma

    """

    def __init__(self,
                 alpha: tuple,
                 image_size: tuple,
                 A: pylops.LinearOperator,
                 lam: Union[float, np.ndarray] = 1,
                 tau: float = 1/np.sqrt(12),
                 data = None,
                 reg_mode = 'tv',
                 silent_mode: bool = True,
                 gamma: Union[float, bool] = False):

        self.lam = lam
        self.A = A
        self.alpha = alpha
        self.tau = tau
        self.tol = 1e-4
        self.max_iter = 150
        self.x = 0
        self.image_size = image_size
        self.sens = 100
        self.gamma = False
        self.silent_mode = silent_mode

        self.data = data
        self.reg_mode = reg_mode

        if False:
            self.breg_p = np.zeros(A.image_dim).ravel()
        else:
            self.breg_p = 0

    def solve(self, f: np.ndarray):
        """
        Description of main primal-dual iteration.
        :return: None
        """

        (primal_n, primal_m) = self.image_size

        v = w = 0
        g = f.ravel()
        p = p_bar = np.zeros(primal_n*primal_m)
        q = q_bar = np.zeros(2*primal_n*primal_m)

        if self.reg_mode != 'tik':
            grad = pylops.Gradient(dims=(primal_n, primal_m), dtype='float64', edge=True, kind="backward")
        else:
            grad = pylops.Identity(np.prod(self.image_size))

        grad1 = pylops.BlockDiag([grad, grad])  # symmetric dxdy <-> dydx not necessary (expensive) but easy and functional

        proj_0 = IndicatorL2((primal_n, primal_m), upper_bound=self.alpha[0])
        proj_1 = IndicatorL2((2 * primal_n, primal_m), upper_bound=self.alpha[1])

        if not self.silent_mode:
            progress = progressbar.ProgressBar(max_value=self.max_iter)

        k = 0

        while (self.tol < self.sens or k == 0) and (k < self.max_iter):

            p_old = p
            q_old = q

            # Dual Update
            g = self.lam / (self.tau + self.lam) * (g + self.tau * (self.A*(p_bar ) - f)) #- self.alpha[0]*self.breg_p

            if self.reg_mode != 'tik':
                v = proj_0.prox(v + self.tau * (grad * p_bar - q_bar))
            else:
                v = self.alpha[0] / (self.tau + self.alpha[0]) * \
                    (v + self.tau * (grad*p_bar - self.data))


            if self.reg_mode == 'tgv':
                w = proj_1.prox(w + self.tau * grad1*q_bar)

            # Primal Update
            p = p - self.tau * (-self.alpha[0]*self.breg_p + self.A.H*g + grad.H * v)


            if self.reg_mode == 'tgv':
                q = q + self.tau * (v - grad1.H * w)

            # Extragradient Update

            p_bar = 2 * p - p_old
            q_bar = 2 * q - q_old


            if k % 50 == 0:
                p_gap = p - p_old
                self.sens = np.linalg.norm(p_gap)/np.linalg.norm(p_old)
                print(self.sens)

            if self.gamma:
                raise NotImplementedError("The adjustment of the step size in the "
                                          "Primal-Dual is not yet fully developed.")
                thetha  = 1 / np.sqrt(1 + 2*self.gamma * self.G.prox_param)
                self.G.prox_param = thetha * self.G.prox_param
                self.F_star.prox_param = self.F_star.prox_param / thetha

            k += 1
            if not self.silent_mode:
                progress.update(k)

        self.x = p

        if k <= self.max_iter:
            print(" Early stopping.")

        return self.x
