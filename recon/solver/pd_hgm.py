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

from recon.terms import BaseDataterm, BaseRegTerm


class PdHgm(object):
    """
    Primal Dual Solver.

    argmin G(x) + F(Kx)

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

    def __init__(self, K: pylops.LinearOperator, F_star: BaseRegTerm, G: BaseDataterm,
                 gamma: Union[float, bool] = False, silent_mode: bool = True):

        self.K = K
        self.F_star = F_star
        self.G = G
        self.max_iter = 2000
        self.gamma = gamma
        self.tol = 1e-4
        self.k = 1
        self.sens = 0.001
        self.silent_mode = silent_mode

        self.plot_on = False

        self.x = self.x_prev = None
        self.y = self.y_prev = None

    def initialise(self, primal_dual = ()):
        """
        Set default start params.
        :return: None
        """
        self.k = 0
        self.res = np.inf
        self.resold = 1
        self.sens = 0.001
        if primal_dual:
            self.x = primal_dual[0]
            self.y = primal_dual[1]
        else:
            self.x = np.zeros((self.K.shape[1]))
            self.y = np.zeros((self.K.shape[0]))
        return

    def solve(self):
        """
        Description of main primal-dual iteration.
        :return: None
        """
        if self.x is None and self.y is None:
            self.initialise()

        if self.plot_on:
            raise NotImplementedError()

        if not self.silent_mode:
            progress = progressbar.ProgressBar(max_value=self.max_iter)

        while (self.tol < self.sens or self.k == 0) and (self.k < self.max_iter):

            self.x_prev = copy.copy(self.x)
            self.y_prev = copy.copy(self.y)

            # primal iteration
            self.x = self.G.prox(self.x - self.G.prox_param * (self.K.H * self.y))

            # dual iteration
            self.y = self.F_star.prox(self.y + self.F_star.prox_param *
                                                (self.K * (2 * self.x - self.x_prev))
                                      )

            if self.k % 100 == 0:
                self.update_sensivity()

            if self.gamma:
                raise NotImplementedError("The adjustment of the step size in the "
                                          "Primal-Dual is not yet fully developed.")
                thetha  = 1 / np.sqrt(1 + 2*self.gamma * self.G.prox_param)
                self.G.prox_param = thetha * self.G.prox_param
                self.F_star.prox_param = self.F_star.prox_param / thetha

            self.k += 1
            if not self.silent_mode:
                progress.update(self.k)

        if self.k <= self.max_iter:
            print(" Early stopping.")

        return self.x

    def update_sensivity(self, quality: str = "high"):
        """
        Update for sensivity

        """
        x_gap = self.x - self.x_prev
        y_gap = self.y - self.y_prev
        if quality == "high":
            self.sens = 1 / 2 * (np.linalg.norm(x_gap - self.G.prox_param * (self.K.H * y_gap), 2) /
                                 np.linalg.norm(self.x, 2) +
                                 np.linalg.norm(y_gap - self.F_star.prox_param * (self.K * x_gap), 2) /
                                 np.linalg.norm(self.y, 2))
        elif quality == "low":
            self.sens = np.linalg.norm(x_gap, 2)/np.linalg.norm(self.x, 2)

        return
