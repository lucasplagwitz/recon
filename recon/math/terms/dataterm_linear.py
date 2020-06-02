import numpy as np

from recon.helpers.objects import dimensions

from scipy import sparse

class DatatermLinear(object):
    """
    Aquivalend to dataterm without operator:

    1

    """

    def __init__(self):
        """

        :param S: sampling Matrix for
        :param F: Operator
        """
        self.tau = 1
        self.proxdata = None
        #self.diagS = (S.T*S).diagonal()


    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def set_proxdata(self, proxdata):
        self.proxdata = proxdata #.T #.ravel()

    def prox(self, f):
        """
        u = F_star * (( F*f + tau * S_star*f_0) / (1 + tau * S_star*S)
        u = f + phi * A_star * f_0 /
        :param f:
        :return:
        """
        u = (f +  self.get_proxparam() * self.proxdata) / (
            1 + self.get_proxparam())
        return u