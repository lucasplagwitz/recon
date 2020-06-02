import numpy as np

class BaseDataterm(object):

    def __init__(self, F):
        self.dim = F.shape[1]
        self.tau = 1
        self.proxdata = np.zeros(S.shape[0])
        self.F = F

    def modify_data(self):
        self.data = (self.S.T)*self.proxdata

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def set_proxdata(self, proxdata):
        self.proxdata = proxdata.T.ravel()
        self.modify_data()

    def prox(self, f):
        u = self.F.T*((self.F*f + self.get_proxparam() * (self.data)) / (
            1 + self.tau * self.diagS))
        return u