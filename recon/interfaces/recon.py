from typing import Union
import numpy as np

from recon.terms import DatanormL2
from recon.solver import PdHgm
from recon.interfaces import BaseInterface


class Recon(BaseInterface):
    """
    A reconstruction interface to solve regularized inverse problems.
    Solver is Primal-Dual based.
    Form:
        lam/2 * ||A*x - f||^2 + \alpha J(x)

        with Operator A : X -> Y
        J(x) regularisation term
    """

    def __init__(self,
                 operator,
                 domain_shape: Union[np.ndarray, tuple],
                 reg_mode: str = '',
                 alpha: float = 0.01,
                 lam: float = 1,
                 tau: Union[float, str] = 'auto',
                 sampling: Union[np.ndarray, None] = None):

        assert self._check_operator(operator)

        super(Recon, self).__init__(domain_shape=domain_shape,
                                    reg_mode=reg_mode,
                                    alpha=alpha,
                                    possible_reg_modes=['tv', 'tikhonov', None],
                                    tau=tau)

        self.G = DatanormL2(operator=operator, image_size=domain_shape, prox_param=self.tau, lam=lam, sampling=sampling)

    def solve(self, data: np.ndarray, max_iter: int = 1000, tol: float = 1e-4):

        super(Recon, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()
        self.solver = PdHgm(self.K, self.F_star, self.G)
        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        return np.reshape(self.solver.x, self.domain_shape)

    @staticmethod
    def _check_operator(operator):
        if hasattr(operator, 'inv'):
            return True
        msg = "Recon expected an operator with .H property as adjoint. Pleas check out the documentation."
        raise ValueError(msg)
