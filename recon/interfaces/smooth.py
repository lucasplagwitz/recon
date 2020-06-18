from pylops import Diagonal
import numpy as np

from recon.terms import DatatermLinear
from recon.solver.pd_hgm import PdHgm
from recon.interfaces import BaseInterface


class Smooth(BaseInterface):
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

        super(Smooth, self).__init__(domain_shape=domain_shape,
                                     reg_mode=reg_mode,
                                     possible_reg_modes=['tv', 'tikhonov', None],
                                     alpha=0, # got possibilty of local param
                                     tau=tau)


        self.local_param = False

        if not isinstance(alpha, float):
            if self.alpha.shape == self.domain_shape:
                self.local_param = True
            else:
                msg = "shape of local parameter alpha does not match: "+ \
                      str(self.alpha.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)

        self.alpha = alpha

        self.G = DatatermLinear()

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        super(Smooth, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.set_proxdata(data.ravel())
        self.solver = PdHgm(self.K, self.F_star, self.G)
        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        return np.reshape(self.solver.var['x'], self.domain_shape)





