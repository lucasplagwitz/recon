from typing import Union
import numpy as np
from pylops import Gradient, BlockDiag, Diagonal

from recon.terms import DatanormL2, IndicatorL2


class BaseInterface(object):
    """
    A base class for primal-dual intercace.
    Handle general input params and shapes.

    Base class for problems with form:
        argmin_u G(u) + F(K(u))

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
    * tau: Union[float, {'auto', 'calc'}]
        Primal-Dual variable, dependent on K - tau \in (0, ||K||).
        First not distingusih between sigma and tau in the interface.

    """

    def __init__(self,
                 domain_shape: np.ndarray,
                 image_shape: np.ndarray = None,
                 reg_mode: str = '',
                 possible_reg_modes: list = None,
                 alpha: float = 1,
                 lam: float = 1,
                 tau: float = None):


        self._reg_mode = None
        self._alpha = None
        self._lam = None
        self.local_alpha = False  # local regularization of regularization term
        self.local_lam = False    # local regularization of data fidelity

        self.possible_reg_modes = possible_reg_modes
        self.domain_shape = domain_shape

        self.reg_mode = reg_mode

        # weights
        self.lam = lam
        self.alpha = alpha

        self.solver = None
        self.tau = 1  # tmp
        self.set_up_operator()

        if isinstance(tau, (float, int)):
            self.tau = tau
        elif tau == 'auto':
            self.tau = 1/np.sqrt(12)  # see references - only 2d TGV
        elif tau == 'calc':
            self.tau = self.calc_tau()
        elif tau == 'relax':
            raise NotImplementedError("relax is not implemented yet")
            sk = self.K.tosparse()
            t = 1/sk.sum(axis=1)
            self.tau = Diagonal(t)
        else:
            msg = "expected tau to be int, float or in ['calc', 'auto']."

        self.solver = None
        self.set_up_operator()

        self.K = None

    def calc_tau(self) -> float:
        norm = np.abs(np.asscalar((self.K.H*self.K).eigs(neigs=1, symmetric=True, largest=True, uselobpcg=True)))
        fac = 0.99
        return fac * np.sqrt(1 / norm)

    def set_up_operator(self) -> None:
        assert self.reg_mode in self.possible_reg_modes

        # since every interface solves via Gradient at this time - will change in future versions
        self.K = Gradient(self.domain_shape, edge=True, dtype='float64', kind='backward', sampling=1)
        #if self.local_alpha:
        #    self.K = BlockDiag([Diagonal(self.alpha.ravel())]*len(self.domain_shape)) * self.K

        if self.reg_mode == 'tv':
            self.F_star = IndicatorL2(self.domain_shape,
                                      len(self.domain_shape),
                                      prox_param=self.tau,
                                      upper_bound=self.alpha)
        elif self.reg_mode == 'tikhonov':
            self.K = self.alpha*self.K  # it is possible to rewrite DatanormL2 -> x/self.alpha and lam=self.alpha
            self.F_star = DatanormL2(image_size=self.K.shape[0], data=0, prox_param=self.tau)
        else:
            self.F_star = DatanormL2(image_size=self.domain_shape, data=0, prox_param=self.tau)
            self.K = 0
        return

    def solve(self, data: np.ndarray, max_iter: int = 5000, tol: float = 1e-5):
        self.set_up_operator()

    @property
    def reg_mode(self):
        return self._reg_mode

    @reg_mode.setter
    def reg_mode(self, value):
        if value in self.possible_reg_modes:
            self._reg_mode = value
        else:
            msg = "Please use reg_mode out of " + str(self.possible_reg_modes)
            raise ValueError(msg)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not (isinstance(value, (int, float, tuple))):
            if value.shape == self.domain_shape:
                self.local_alpha = True
            else:
                msg = "shape of local parameter alpha does not match: " + \
                      str(self.alpha.shape) + "!=" + str(self.domain_shape)
                raise ValueError(msg)
        self._alpha = value

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, value):
        if not (isinstance(value, (int, float))):
            if value.shape == self.domain_shape or \
                    (len(value.shape) == 1 and value.shape == np.prod(self.domain_shape)):
                self.local_lam = True
                value = value.ravel()
            else:
                msg = "shape of local parameter alpha does not match: " + \
                      str(value.shape) + "!=" + str(self.domain_shape)
                raise ValueError(msg)
        self._lam = value
