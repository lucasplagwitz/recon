from typing import Union
import numpy as np

from recon.terms import DatanormL1, DatanormL2
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
                 domain_shape: Union[np.ndarray, tuple],
                 reg_mode: str = '',
                 alpha: float = 1,
                 lam: float = 1,
                 norm: str = 'L2',
                 tau: Union[float, str] = 'calc'):

        super(Smoothing, self).__init__(domain_shape=domain_shape,
                                        reg_mode=reg_mode,
                                        possible_reg_modes=['tv', 'tikhonov', None],
                                        alpha=alpha,
                                        lam=lam,
                                        tau=tau)

        if norm == 'L2':
            self.G = DatanormL2(domain_shape, lam=self.lam, prox_param=self.tau)
        elif norm == 'L1':
            self.G = DatanormL1(domain_shape, lam=self.lam, prox_param=self.tau)

    def solve(self, data: np.ndarray, max_iter: int = 5000, tol: float = 1e-4):
        super(Smoothing, self).solve(data=data, max_iter=max_iter, tol=tol)
        self.G.data = data.ravel()

        self.solver = PdHgm(self.K, self.F_star, self.G)
        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        return np.reshape(self.solver.x, self.domain_shape)
