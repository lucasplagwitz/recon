import numpy as np

from recon.terms import BaseDataterm


class DatatermBregmanSegmentation(BaseDataterm):

    def __init__(self, O, A, B, sampling=None):
        super(DatatermBregmanSegmentation, self).__init__(O=O, sampling=sampling)

        self.A = A
        self.B = B
        self.pk = np.zeros(O.shape[1])
        self.alpha = 0

    def get_proxparam1(self):
        return self.alpha

    def set_proxparam1(self, alpha):
        self.alpha = alpha


    def prox(self, f):

        u = self.O.H*(
                (self.O * (f + self.get_proxparam() * self.B + self.get_proxparam() * self.get_proxparam1() * self.pk) +
                    self.get_proxparam() * self.data) /
                (1 + self.tau * self.diag_sampling + self.get_proxparam() * self.A)
        )
        return u

    def setP(self, pk):
        self.pk = pk.ravel()

    def getP(self):
        return self.pk
