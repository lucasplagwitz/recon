from typing import Union
from recon.operator import Identity
import numpy as np

from recon.terms.base_term import BaseDataterm


class DatanormL1(BaseDataterm):
    """Datanorm-L1

    This class is the basic form of the L2-Datanorm in terms of A: X -> Y.

    Function u(x):
        lambda/2 * ||Ax - f||_1

    Special Form with A=identity:
        lambda/2 * ||x - f||_1

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
                 operator=None,
                 data: Union[float, np.ndarray] = 0,
                 lam: float = 1,
                 prox_param: float = 0.9,
                 sampling=None):
        if operator is None:
            operator = Identity(domain_dim=image_size)
        else:
            raise NotImplementedError("Currently not supported.")

        super(DatanormL1, self).__init__(operator, sampling=sampling, prox_param=prox_param)
        self.lam = lam
        self.data = data
        self.inv_operator = self.operator.inv

    def __call__(self, x):
        return self.lam*np.abs(self.operator*x-self.data)

    def prox(self, x):
        """
        Proximal Operator

        prox(x) = A_inv * (( A*x + tau * lambda * f) / (1 + tau * lambda * diag_sampling))
        """
        diff = x - self.data
        u = 0 + (diff > (self.prox_param * self.lam)).astype(int)*(x-self.prox_param*self.lam) + \
               (diff < -(self.prox_param * self.lam)).astype(int)*(x+self.prox_param*self.lam) + \
               (np.abs(diff) <= (self.prox_param * self.lam)).astype(int) * self.data

        return u
