from pylops import LinearOperator, Gradient
import numpy as np

from recon.math.terms import Dataterm, Projection, DatatermLinear
from recon.math.pd_hgm import PdHgm

class PdRecon(object):
    """
    A Reconstruction object to solve regularized inverse reconstruction problems.
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
        self._reg_mode = None

        self.O = O
        self.domain_shape = domain_shape
        self.alpha = alpha
        self.tau = tau
        self.reg_mode = reg_mode
        self.solver = None


    @property
    def reg_mode(self):
        return self._reg_mode

    @reg_mode.setter
    def reg_mode(self, value):
        if value in ['tikhonov', 'tv', None]:
            self._reg_mode = value
        else:
            msg = "Please use reg_mode out of ['tikhonov', 'tv', '']"
            raise ValueError(msg)

    def solve(self, f: np.ndarray, maxiter: int = 150, tol: float = 5*10**(-4)):

        if self.reg_mode is not None:
            grad = Gradient(self.domain_shape, dtype='float64')
            K = self.alpha * grad

            if not self.tau:
                norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                sigma = 0.99 / norm
                tau = sigma
            else:
                tau = self.tau
                sigma = tau

            if self.reg_mode == 'tv':
                F_star = Projection(self.domain_shape, len(self.domain_shape))
            else:
                F_star = DatatermLinear()
        else:
            tau = 0.99
            sigma = tau
            F_star = DatatermLinear()
            K = 0

        G = Dataterm(self.O)
        G.set_proxparam(tau)
        F_star.set_proxparam(sigma)
        self.solver = PdHgm(K, F_star, G)
        self.solver.maxiter = maxiter
        self.solver.tol = tol

        G.set_proxdata(f.ravel())
        self.solver.solve()

        return np.reshape(self.solver.var['x'], self.domain_shape)



