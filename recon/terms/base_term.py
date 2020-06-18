import pylops
import numpy as np

class BaseDataterm(object):

    def __init__(self, O, sampling=None):
        self.tau = 0.99
        self.O = O
        self.data = None
        self.samling = sampling
        if not sampling:
            self.sampling = 1
            self.diag_sampling = 1
            self.sampling_transpose = 1
        else:
            self.diag_sampling = (self.sampling.T * self.sampling).diagonal()
            self.sampling_transpose = self.samling.T

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def set_proxdata(self, proxdata):
        if isinstance(proxdata, (int, float)):
            self.data = proxdata
        else:
            self.data = (self.sampling_transpose)*proxdata

    def prox(self, f):
        """
        proximal operator of term
        """
        pass


class BaseRegTerm(object):

    def __init__(self):
        self.sigma = 0.99

    def set_proxparam(self, sigma):
        self.tau = sigma

    def get_proxparam(self):
        return self.sigma

    def prox(self, f):
        """
        proximal operator of term
        """
        pass