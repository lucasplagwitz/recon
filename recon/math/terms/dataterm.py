import numpy as np
from scipy import sparse

class Dataterm(object):

    def __init__(self, S, F):
        self.dim = F.shape[1]
        self.tau = 1
        self.proxdata = np.zeros(S.shape[0])
        self.S = S
        self.diagS = (S.T*S).diagonal()
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
        c= self.F.T*((self.F*f + self.get_proxparam() * (self.data)))
        u = self.F.T*((self.F*f + self.get_proxparam() * (self.data)) / (
            1 + self.tau * self.diagS))
        return u