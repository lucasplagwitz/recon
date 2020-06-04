import numpy as np
import pylops

class DatatermRecBregman(object):

    def __init__(self, S, F):
        #self.dim = F.shape_nd[1]
        self.tau = 1
        if S is not None:
            self.proxdata = np.zeros(S.shape[0])
            self.S = S
            self.diagS = (S.T * S).diagonal()
        else:
            self.diagS = 1
            self.S = None
        self.F = F
        #self.pk = np.zeros(F.shape_nd[1])
        self.alpha = 1

        if isinstance(F, pylops.LinearOperator):
            self.pylops = True

    def modify_data(self):
        if self.S is not None:
            self.data = (self.S.T) * self.proxdata
        else:
            self.data = self.proxdata

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def get_proxparam1(self):
        return self.alpha

    def set_proxparam1(self, alpha):
        self.alpha = alpha

    def set_proxdata(self, proxdata):
        self.proxdata = proxdata.T.ravel()
        self.modify_data()

    def prox(self, f):
        if self.pylops:
            F_star = self.F.H
        else:
            F_star = self.F.T
        u = F_star*((self.F * (f+ self.get_proxparam() * self.get_proxparam1() * self.pk) +
                       self.get_proxparam() * self.data) / (
                        1 + self.tau * self.diagS))
        return u

    def setP(self, pk):
        self.pk = pk.T.ravel()

    def getP(self):
        return self.pk