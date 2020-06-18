from pylops import LinearOperator
import numpy as np

from recon.terms import Dataterm
from recon.solver import PdHgm
from recon.interfaces import BaseInterface

class Recon(BaseInterface):
    """
    A Reconstruction object interface to solve regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        1/2 * ||O*x - f||^2 + \alpha J(x)

        with Operator O : X -> Y
        J(x) regularisation term
    """

    def __init__(self,
                 O: LinearOperator,
                 domain_shape: np.ndarray,
                 reg_mode: str = '',
                 alpha: float= 0.01,
                 tau: float = None):

        super(Recon, self).__init__(domain_shape=domain_shape,
                                    reg_mode=reg_mode,
                                    alpha=alpha,
                                    possible_reg_modes=['tv', 'tikhonov', None],
                                    tau=tau)

        self.O = O
        self.G = Dataterm(self.O)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        super(Recon, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.set_proxparam(self.tau)
        self.G.set_proxdata(data.ravel())
        self.solver = PdHgm(self.K, self.F_star, self.G)
        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        return np.reshape(self.solver.var['x'], self.domain_shape)



