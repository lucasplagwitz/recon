from pylops.basicoperators import Gradient
from pylops import LinearOperator
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
                 alpha: float= 0.01,
                 tau: float = None):
        self._reg_mode = None

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

    def solve(self, data: np.ndarray, maxiter: int = 150, tol: float = 5*10**(-4)):

        if self.reg_mode is not None:
            if len(self.domain_shape)>2:
                grad = Gradient(dims=self.domain_shape, edge = True, dtype='float64')
            else:
                ex = np.ones((self.domain_shape[1], 1))
                ey = np.ones((1, self.domain_shape[0]))
                dx = sparse.diags([1, -1], [0, 1], shape=(self.domain_shape[1], self.domain_shape[1])).tocsr()
                dx[self.domain_shape[1] - 1, :] = 0
                dy = sparse.diags([-1, 1], [0, 1], shape=(self.domain_shape[0], self.domain_shape[0])).tocsr()
                dy[self.domain_shape[0] - 1, :] = 0

                grad = sparse.vstack((sparse.kron(dx, sparse.eye(self.domain_shape[0]).tocsr()),
                                  sparse.kron(sparse.eye(self.domain_shape[1]).tocsr(), dy)))

            K = self.alpha * grad
            if not self.tau:
                if np.prod(self.domain_shape) > 25000:
                    long = True
                else:
                    long = False
                if long:
                    print("Start evaluate tau. Long runtime.")
                if len(self.domain_shape)>2:
                    norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                else:
                    norm = normest(K)
                sigma = 0.99 / norm
                if long:
                    print("Calc tau: "+str(sigma))
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



