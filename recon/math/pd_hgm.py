"""
Primal Dual Algorithm based on Paper:
A first-order primal-dual algorithm for convex problems with applications to imaging
by Antonin Chambolle, Thomas Pock.

Implementation is based on Veronica Corona et al.
see: https://github.com/lucasplagwitz/recon/blob/master/recon/math/pd_hgm.py

For reading some variables are renamed and for more mathematical details of Primal Dual
see abstract /documentation/primal_dual.pdf
"""
import numpy as np
import copy
import sys
import pylops

class PdHgm(object):
    """
    Primal Dual Solver.

    Algorithm: (See /documentation/primal_dual.pdf for derivation)
            while tol < sens and maxiter not reached
                1. x^{n+1} = prox_{G}( x^{n} - phi * (K.T * y^{n}) )
                2. y^{n+1} = prox_{F_star}( y^{n} - sigma * (K * (2*x^{n+1} - x^{n})) )
                3. update sens
    """

    def __init__(self, K, F_star, G, param_alpha = None):
        """
        Consturctor. Set required params.
        :param K:
        :param F_star:
        :param G:
        """
        self.K = K
        self.F_star = F_star
        self.G = G
        self.maxiter = 500
        self.tol = 10**(-4)
        self.k = 1
        self.sens = 0.001

        self.plot_on = False

        self.var = {'x': None, 'y': None}

        if param_alpha is not None:
            self.param_alpha = param_alpha

        if isinstance(K, pylops.LinearOperator):
            self.pylops = False

    def restart_counter(self):
        """
        Reset iteration counter.
        :return: None
        """
        self.k = 1
        return

    def initialise(self, primal_dual = ()):
        """
        Set default start params.
        :return: None
        """
        self.k = 1
        self.res = np.inf
        self.resold = 1
        self.sens = 0.001
        if primal_dual:
            self.var['x'] = primal_dual[0]
            self.var['y'] = primal_dual[1]
        else:
            self.var['x'] = np.zeros((self.K.shape[1]))
            self.var['y'] = np.zeros((self.K.shape[0]))
        return

    def solve(self):
        """
        Description of main primal-dual iteration.
        :return: None
        """
        if self.var['x'] is None and self.var['y'] is None:
            self.initialise()

        if self.plot_on:
            raise NotImplementedError()

        # setup toolbar
        toolbar_limit = 40
        toolbar_current = 5
        sys.stdout.write("Primal-Dual Algorithm: ")
        sys.stdout.write("[%s]" % (" " * toolbar_limit))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_limit + 1))

        while (self.tol < self.sens or self.k == 1) and (self.k <= self.maxiter):

            self.var_prev = copy.copy(self.var)

            # primal iteration
            self.var['x'] = self.G.prox(self.var['x'] -
                                        self.G.get_proxparam() * (self.K.T * self.var['y'])
                                        )

            # dual iteration
            self.var['y'] = self.F_star.prox(self.var['y'] +
                                             self.F_star.get_proxparam() *
                                                (self.K * (2 * self.var['x'] - self.var_prev['x']))
                                            )

            self.update_sensivity()

            self.k += 1

            if ((self.k / self.maxiter) * 100) >= toolbar_current:
                sys.stdout.write("-")
                sys.stdout.flush()
                toolbar_current += 5

        if self.k <= self.maxiter:
            sys.stdout.write("]\n" + "early stopping!")
        else:
            sys.stdout.write("]\n")

        return None

    def update_sensivity(self):
        """
        Update for sensivity

        Formula: (See /documentation/primal_dual.pdf for derivation)
            x_gap^{n+1} = x^{n+1} - x^{n}
            y_gap^{n+1} = y^{n+1} - y^{n}
            sens^{n+1} = 1/2 * ( || x_gap^{n+1} - phi * (K.T * y_gap) || / ||x^{n+1}|| +
                                 || y_gap^{n+1} - sigma * (K * x_gap) || / ||y^{n+1}||
                               )

        :return: None
        """

        x_gap = self.var['x'] - self.var_prev['x']
        y_gap = self.var['y'] - self.var_prev['y']
        self.sens = 1 / 2 * (np.linalg.norm(x_gap - self.G.get_proxparam() * (self.K.T * y_gap), 2) /
                             np.linalg.norm(self.var['x'], 2) +
                             np.linalg.norm(y_gap - self.F_star.get_proxparam() * (self.K * x_gap), 2) /
                             np.linalg.norm(self.var['y'], 2))
        return
