from typing import Union
import numpy as np

from recon.terms import DatanormL2
from recon.solver.pd_hgm import PdHgm
from recon.interfaces import BaseInterface


class Smoothing(BaseInterface):
    """
    A smoothing interface based on regularization techniques.
    Form A+B with A near to input data and B
    Solver is Primal-Dual based.

    Form:
        lambda/2 * ||x - f||^2 + \alpha J(x)

        J(x) regularization term J in [TV(), L2(Grad())]
    """

    def __init__(self,
                 domain_shape: np.ndarray,
                 reg_mode: str = '',
                 alpha:float = 0.01,
                 lam: float = 1,
                 tau: Union[float, str] = 'auto'):

        super(Smoothing, self).__init__(domain_shape=domain_shape,
                                        reg_mode=reg_mode,
                                        possible_reg_modes=['tv', 'tikhonov', None],
                                        alpha=alpha,
                                        lam=lam,
                                        tau=tau)

        self.G = DatanormL2(domain_shape, lam=lam, data=0)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):
        super(Smoothing, self).solve(data=data, max_iter=max_iter, tol=tol)
        self.G.data = data.ravel()

        self.solver = PdHgm(self.K, self.F_star, self.G)
        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        return np.reshape(self.solver.var['x'], self.domain_shape)
