from pylops.basicoperators import Gradient
from pylops import LinearOperator, Diagonal, VStack, BlockDiag
import numpy as np
from scipy import sparse

from recon.math.terms import Dataterm, Projection, DatatermLinear
from recon.math.pd_hgm import PdHgm
from recon.helpers.functions import normest


class PdSmooth(object):
    """
    A Reconstruction object to solve regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        1/2 * ||x - f||^2 + \alpha J(x)

        J(x) regularisation term J in [TV(), || ||]
    """

    def __init__(self,
                 domain_shape: np.ndarray,
                 reg_mode: str = '',
                 alpha=0.01,
                 tau: float = None):
        self._reg_mode = None

        self.domain_shape = domain_shape
        self.alpha = alpha
        self.tau = tau
        self.reg_mode = reg_mode
        self.solver = None
        self.local_param = False

        if type(alpha) is not float:
            if self.alpha.shape == self.domain_shape:
                self.alpha = Diagonal(self.alpha.ravel())
                self.local_param = True
            else:
                msg = "shape of local parameter alpha does not match: "+ \
                      str(self.alpha.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)

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

    def solve(self, data: np.ndarray, maxiter: int = 150, tol: float = 5*10**(-4)):

        if self.reg_mode is not None:
            grad = Gradient(dims=self.domain_shape, edge = True, dtype='float64', kind='backward')
            if self.local_param:
                K = BlockDiag([self.alpha]*len(self.domain_shape)) * grad
            else:
                K = self.alpha * grad

            if not self.tau:
                norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                sigma = 0.99 / norm
                print("Calced tau: " + str(sigma) + ". "
                      "Next run with same alpha: set this tau value to decrease runtime.")
                tau = sigma
            else:
                tau = self.tau
                sigma = tau

            if self.reg_mode == 'tv':
                F_star = Projection(self.domain_shape, len(self.domain_shape))
            else:
                F_star = DatatermLinear()
                F_star.set_proxdata(0)
        else:
            tau = 0.99
            sigma = tau
            F_star = DatatermLinear()
            K = 0

        G = DatatermLinear()
        G.set_proxparam(tau)
        G.set_proxdata(data.ravel())
        F_star.set_proxparam(sigma)
        self.solver = PdHgm(K, F_star, G)
        self.solver.maxiter = maxiter
        self.solver.tol = tol
        self.solver.solve()

        return np.reshape(self.solver.var['x'], self.domain_shape)



