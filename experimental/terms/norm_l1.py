from typing import Union
from pylops import Identity
import numpy as np

from recon.terms.base_term import BaseDataterm

class NormL1(BaseDataterm):
    """NormL1

    This class is the basic form of the L2-Datanorm in terms of A: X -> Y.

    Function u(x):
        lambda * ||Ax - f||_1

    Special Form 1. with A=identity:
        lambda * ||x - f||_1

    Special Form 2. with A=identity and f=0:
        lambda * ||x||_1

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
            operator = Identity(N=np.prod(image_size))

        super(NormL1, self).__init__(operator, sampling=sampling)
        self.lam = lam
        self.prox_param = prox_param
        self.data = data

    def __call__(self, x):
        return self.lam*np.sum(np.abs(self.operator*x-self.data))

    def prox(self, x):
        raise NotImplementedError("The proximal Operator is not implemented yet.")