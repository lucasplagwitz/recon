import numpy as np
from pylops import Gradient, BlockDiag
import matplotlib.pyplot as plt

from recon.terms import IndicatorL2, DatanormL2, DatanormL2Bregman

class PdHgmTGV(object):
    """
    Primal Dual Solver for pairwise primal and dual variable.

    Implementation based on:
        Knoll, Bredis, Pock: Second Order Total Generalized Variation (TGV) for MRI

    K1 := div_1     K1star=-grad
    K2 := div_2     K2star=-epsilon

    Algorithm: dual variable (p,q) - primal variable (u, v)
            while tol < sens and max_iter not reached
                1. p^{n+1} = prox_F1^star(p^{n} + sigma*(K_1*u - v))
                2. q^{n+1} = prox_F2^star()

                3. (u_old, v_old) = (u, v)
                4. u = 2*Prox_G1(u+tau*)


                1. x^{n+1} = prox_{G}( x^{n} - phi * (K.T * y^{n}) )
                2. y^{n+1} = prox_{F_star}( y^{n} - sigma * (K * (2*x^{n+1} - x^{n})) )
                3. update sens
    """
    def __init__(self,
                 lam: float = 1,
                 alpha: tuple = (1, 1),
                 tol=1e-4,
                 mode: str = 'tv',
                 pk: np.ndarray = None,
                 prox_param: float = 1/np.sqrt(12)):
        """
        Consturctor. Set required params.
        """
        self.lam = lam
        self.max_iter = 3000
        self.alpha = alpha   # form: (a_1, a_2)
        self.sigma = prox_param
        self.tau = self.sigma
        self.tol = tol
        self.k = 1
        self.mode = mode
        self.pk = pk  # only for Bregman

    # Todo: symmetric saves double dxdy <-> dydx...not necessary

    def solve(self, f: np.ndarray):
        self.k = 1
        if len(np.shape(f)) != 2:
            raise ValueError("The TGV-Algorithm only implemnted for 2D images. Please give input shaped (m, n)")
        (primal_n, primal_m) = np.shape(f)
        grad = Gradient(dims=(primal_n, primal_m), dtype='float64', edge=True, kind="backward")
        grad_v = BlockDiag([grad, grad])  # symmetric dxdy <-> dydx not necessary (expensive) but easy and functional
        p, q = 0, 0
        v = v_bar = np.zeros(2*primal_n*primal_m)
        u = u_bar = f.ravel()

        # Projections
        proj_p = IndicatorL2((primal_n, primal_m), upper_bound=self.alpha[0])
        proj_q = IndicatorL2((2*primal_n, primal_m), upper_bound=self.alpha[1])
        if self.mode == 'tv':
            dataterm = DatanormL2(image_size=f.shape, data=f.ravel(), prox_param=self.tau, lam=self.lam)
        else:
            dataterm = DatanormL2Bregman(image_size=f.shape, data=f.ravel(), prox_param=self.tau, lam=self.lam)
            dataterm.pk = self.pk
            dataterm.bregman_weight_alpha = self.alpha[0]
        sens = 100
        while (self.tol < sens or self.k == 1) and (self.k <= self.max_iter):
            p = proj_p.prox(p + self.sigma*(grad*u_bar - v_bar))
            q = proj_q.prox(q + self.sigma*(grad_v*v_bar)) #self.adjoint_div(v_bar, 1)
            u_old = u
            v_old = v
            u = dataterm.prox(u - self.tau*grad.H*p)
            u_bar = 2*u - u_old
            v = v + self.tau*(p - grad_v.H*q)
            v_bar = 2*v - v_old

            #self.update_sensivity(u, u_old, v, v_old, grad)
            # test
            if self.k % 300 == 0:
                u_gap = u-u_old
                v_gap = v-v_old
                sens = 1/2*(
                        np.linalg.norm(u_gap - self.tau*grad.H*proj_p.prox(p+self.sigma*(grad*u_gap - v_gap)), 2)/
                        np.linalg.norm(u, 2) +
                        np.linalg.norm(v-proj_p.prox(p+self.sigma*(grad*u_gap - v_gap))-grad_v.H*proj_q.prox(q+ self.sigma*(grad_v*v_gap)), 2)/
                        np.linalg.norm(v, 2))   # not best sens
                print(np.linalg.norm(u_gap, 2))
            self.k += 1
        return np.reshape(u, (primal_n, primal_m))

    def update_sensivity(self, u, u_old, v, v_old, K):
        """
        Update for sensivity

        Formula:
            x_gap^{n+1} = x^{n+1} - x^{n}
            y_gap^{n+1} = y^{n+1} - y^{n}
            sens^{n+1} = 1/2 * ( || x_gap^{n+1} - phi * (K.T * y_gap) || / ||x^{n+1}|| +
                                 || y_gap^{n+1} - sigma * (K * x_gap) || / ||y^{n+1}||
                               )

        :return: None
        """

        u_gap = u - u_old
        v_gap = v - v_old
        #self.sens = 1 / 2 * (np.linalg.norm(u_gap - self.G.get_proxparam() * (self.K.T * y_gap), 2) /
        #                     np.linalg.norm(u, 2) +
        #                     np.linalg.norm(v_gap - self.F_star.get_proxparam() * (self.K * x_gap), 2) /
        #                     np.linalg.norm(v, 2))
        return
