from typing import Union
from pylops import Identity
import numpy as np

from recon.terms import BaseDataterm


class DatanormL2Bregman(BaseDataterm):

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
        self.bregman_weight_alpha = bregman_weight_alpha
        self.image_size = image_size

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