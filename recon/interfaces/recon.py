from typing import Union
import numpy as np

from recon.terms import DatanormL2
from recon.solver import PdHgm, PdHgmExtend
from recon.interfaces import BaseInterface
from recon.utils.utils import power_method


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
                 extend_pdhgm = True,
                 data=None,
                 sampling: Union[np.ndarray, None] = None):

        assert self._check_operator(operator)

        self.operator = operator

        super(Recon, self).__init__(domain_shape=domain_shape,
                                    reg_mode=reg_mode,
                                    lam=lam,
                                    alpha=alpha,
                                    possible_reg_modes=['tv', 'tikhonov', 'tik', 'tgv', None],
                                    tau=tau)

        if hasattr(operator, 'inv') and not extend_pdhgm:
            self.G = DatanormL2(operator=operator, image_size=domain_shape, prox_param=self.tau, lam=lam, sampling=sampling)
            self.extend_pdhgm = False
        else:
            self.data = data
            self.extend_pdhgm = True
            self.operator = operator

            self.norm = power_method(self.operator, self.operator.H, max_iter=100)
            self.tau = 0.99 * np.sqrt(1 / self.norm)
            #self.norm = np.abs(np.asscalar((self.operator.H * self.operator).eigs(neigs=1,
            #                                                                      symmetric=True,
            #                                                                      largest=True,
            #                                                                      uselobpcg=True)))
            #print(self.norm)
            self.breg_p = 0

    def solve(self, data: np.ndarray, max_iter: int = 1000, tol: float = 1e-4):

        super(Recon, self).solve(data=data, max_iter=max_iter, tol=tol)

        if self.extend_pdhgm:
            if isinstance(self.alpha, (float, int)):
                self.alpha = (self.alpha, self.alpha)
            #self.operator.norm = self.norm
            self.solver = PdHgmExtend(A=self.operator,
                                      lam=self.lam,
                                      alpha=self.alpha,
                                      tau=self.tau,
                                      data=data,
                                      image_size=self.domain_shape,
                                      reg_mode=self.reg_mode)
            self.solver.breg_p = self.breg_p
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.solver.solve(data.ravel())
        else:
            self.G.data = data.ravel()
            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.solver.solve()

        return np.reshape(self.solver.x, self.domain_shape)

    @staticmethod
    def _check_operator(operator):
        if hasattr(operator, 'inv') or hasattr(operator, 'H'):
            return True
        msg = "Recon expected an operator with .H property as adjoint. Pleas check out the documentation."
        raise ValueError(msg)

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, value):
        if not (isinstance(value, (int, float))):
            if value.shape == self.operator.image_dim or \
                    (len(value.shape) == 1 and value.shape == np.prod(self.operator.image_dim)):
                self.local_lam = True
                value = value.ravel()
            else:
                msg = "shape of local parameter alpha does not match: " + \
                      str(value.shape) + "!=" + str(self.domain_shape)
                raise ValueError(msg)
        self._lam = value