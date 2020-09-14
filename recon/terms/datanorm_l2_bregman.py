from typing import Union
from pylops import Identity
import numpy as np

from recon.terms import BaseDataterm


class DatanormL2Bregman(BaseDataterm):
    """Datanorm-L2

        This class is the basic form of the L2-Datanorm in terms of A: X -> Y.

        Function u(x):
            lambda/2 * ||Ax - f||_2^2 - w * <p_k, x>

        Special Form with A=identity:
            lambda/2 * ||x - f||_2^2 - w * <p_k, x>

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
        bregman_weight_alpha:
            In equations above the w. Often denoted as alpha.
            Renamed to prevent naming issues with reg-weight alpha.
        sampling:
            Matrix for undersampling data f.
        data: np.ndarry, float
            
        """

    def __init__(self,
                 image_size,
                 operator=None,
                 data: Union[float, np.ndarray] = 0,
                 lam: float = 1,
                 bregman_weight_alpha: float = 1,
                 prox_param: float = 0.9,
                 sampling=None):
        self._pk = None

        if operator is None:
            operator = Identity(N=np.prod(image_size))

        super(DatanormL2Bregman, self).__init__(operator, sampling=sampling)
        self.lam = lam
        self.prox_param = prox_param
        self.data = data
        self.pk = np.zeros(shape=self.data.shape)
        self.bregman_weight_alpha = bregman_weight_alpha
        self.image_size = image_size

    def __call__(self, x):
        return 1/2*np.sqrt(np.sum((self.operator*x-self.data)**2)) - self.bregman_weight_alpha * np.dot(self.pk.T, x)

    def prox(self, f):
        u = self.operator.H*(
                (self.operator * (f + self.prox_param * self.bregman_weight_alpha * self.pk) +
                    self.prox_param * self.data) /
                (1 + self.prox_param * self.diag_sampling))
        return u

    @property
    def pk(self):
        return self._pk

    @pk.setter
    def pk(self, value):
        if len(value.shape) > 1:
            if self.image_size == value.shape:
                self._pk = value.ravel()
                return
        elif self.data.shape == value.shape:
            self._pk = value
            return
        msg = "Something went wrong with the input bregman pk shape."
        raise IndexError(msg)