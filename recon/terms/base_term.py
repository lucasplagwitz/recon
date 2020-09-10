import pylops
import numpy as np

class BaseDataterm(object):

    def __init__(self, operator, sampling=None):
        self._data = None

        self.tau = 0.99
        self.operator = operator
        if sampling is None:
            self.diag_sampling = 1
            self.sampling_transpose = 1
        else:
            self.diag_sampling = (sampling.T * sampling).diagonal()
            self.sampling_transpose = sampling.T

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, (int, float)):
            self._data = value
        elif isinstance(value, np.ndarray) and len(value.shape):
            self._data = (self.sampling_transpose)*value
        else:
            raise ValueError("Require int, float, np.ndarray in raveled form.")


class BaseRegTerm(object):

    def __init__(self, prox_param: float = 0.99):
        self.prox_param = prox_param
