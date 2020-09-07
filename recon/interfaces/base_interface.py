from pylops import LinearOperator, Gradient, BlockDiag, Diagonal
import numpy as np

from recon.terms import Dataterm, Projection, DatatermLinear
from recon.solver.pd_hgm import PdHgm


class BaseInterface(object):
    """
    A base class for primal-dual intercace.
    Handle general input params and shapes.

    PARAMETER
    ---------
    * domain_shape: ndarray
        shape of input
    * reg_mode: str
        mode for regularization (choose between different RegTerms)
    * possible_reg_modes: list
        list of possible regularization modes
    * alpha: float
        weight for regularization effect
    * tau: float
        Primal-Dual variable, dependent on K - tau \in (0, ||K||)
    """

    def __init__(self,
                 domain_shape: np.ndarray,
                 image_shape: np.ndarray = None,
                 reg_mode: str = '',
                 possible_reg_modes: list = None,
                 alpha: float = 0,
                 tau: float = None,
                 weight_term: str = 'reg'):
        self._reg_mode = None

        self.possible_reg_modes = possible_reg_modes
        self.domain_shape = domain_shape
        self.tau = tau
        self.reg_mode = reg_mode
        self.solver = None
        self.alpha = alpha
        self.solver = None
        self.weight_term = weight_term

        self.local_param = False

        if not image_shape:
            self.image_shape = domain_shape
        else:
            self.image_shape = image_shape

        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            if self.alpha.shape == self.domain_shape and self.weight_term == 'reg':
                self.local_param = True
            elif self.alpha.shape == self.image_shape and self.weight_term == 'data':
                self.local_param = True
            else:
                msg = "shape of local parameter alpha does not match: " + \
                      str(self.alpha.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)

        # sovler objects
        self.G = None
        self.F_star = None
        self.K = None


    @property
    def reg_mode(self):
        return self._reg_mode

    @reg_mode.setter
    def reg_mode(self, value):
        if value in self.possible_reg_modes:
            self._reg_mode = value
        else:
            msg = "Please use reg_mode out of "+str(self.possible_reg_modes)
            raise ValueError(msg)

    def calc_tau(self):
        if self.weight_term == 'reg':
            norm = np.abs(np.asscalar(self.K.eigs(neigs=1, which='LM'))) #* self.alpha
        else:
            norm = np.abs(np.asscalar(self.K.eigs(neigs=1, which='LM'))) * np.max(self.alpha)
        return 0.99 / norm

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        # since every interface solves via Gradient at this time - will change in future versions
        if self.reg_mode is not None:
            if self.weight_term == 'reg':
                if self.local_param:
                    self.K = BlockDiag([Diagonal(self.alpha.ravel())]*len(self.domain_shape)) * \
                             Gradient(self.domain_shape, edge=True, dtype='float64', kind='backward')
                else:
                    self.K = self.alpha * Gradient(self.domain_shape, edge = True, dtype='float64', kind='backward')
            else:
                self.K = Gradient(self.domain_shape, edge = True, dtype='float64', kind='backward')

            if not self.tau:
                self.tau = self.calc_tau() #/ 2
                print("Calced tau: " + str(self.tau) + ". "
                      "Next run with same alpha and reg_mode, set the tau param to decrease runtime.")

            if self.reg_mode == 'tv':
                self.F_star = Projection(self.domain_shape, len(self.domain_shape))
            else:
                self.F_star = DatatermLinear()
                self.F_star.set_proxdata(0)
        else:
            self.tau = 0.99
            self.F_star = DatatermLinear()
            self.F_star.set_proxdata(0)
            self.K = 0

        self.F_star.set_proxparam(self.tau)

        return


