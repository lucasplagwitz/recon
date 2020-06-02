import numpy as np

class DatatermLinearRecBregman(object):

    def __init__(self):
        self.tau = 1
        self.proxdata = 0
        self.qk = 0
        self.alpha = 1

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def get_proxparam1(self):
        return self.alpha

    def set_proxparam1(self, alpha):
        self.alpha = alpha

    def set_proxdata(self, proxdata):
        self.proxdata = proxdata

    def prox(self, f):

        u = ((f+ self.get_proxparam() * self.get_proxparam1() * self.qk) +
                       self.get_proxparam() * self.proxdata) / (
                        1 + self.tau)
        return u

    def setQ(self, qk):
        self.qk = qk

    def getQ(self):
        return self.qk