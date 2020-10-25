from typing import Union
from pylops import Identity
import numpy as np

from recon.terms.base_term import BaseRegTerm


class RegnormL2(BaseRegTerm):
    """Regnorm-L2

    This class is the basic form of the L2-Datanorm in terms of A: X -> Y.

    Function u(x):
        alpha * ||x/alpha - f||_2^2

    Parameter
    ---------
    image_size:
        Size of input image in unraveld form.
    operator:
        Should have adjoint method called operator.H.
    lam: float
        weight Parameter, see lambda in above function
    prox_param: float
        same behavior like lam. normally known as tau*u(x)
    sampling:
        Matrix for undersampling data f.
    data: np.ndarry, float
    """

    def __init__(self,
                 image_size,
                 data: Union[float, np.ndarray] = 0,
                 alpha: float = 1,
                 prox_param: float = 0.9):

        super(BaseRegTerm, self).__init__(prox_param=prox_param)
        self.alpha = alpha
        self.data = data

    def __call__(self, x):
        raise NotImplementedError("not implemented")
        return self.lam/2*np.sqrt(np.sum((self.operator*x-self.data)**2))

    def prox(self, x):
        """
        Proximal Operator

        prox(x) = A_inv * (( A*x + tau * lambda * f) / (1 + tau * lambda * diag_sampling))
        """
        u = self.operator.H*(
                (self.operator*x + self.prox_param * self.lam * self.data) /
                (1 + self.prox_param * self.lam * self.diag_sampling)
            )

        return u
