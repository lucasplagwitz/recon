from recon.terms import BaseDataterm


class DatatermBregman(BaseDataterm):

    def __init__(self, O, sampling=None):
        super(DatatermBregman, self).__init__(O, sampling=sampling)
        self.alpha = 1

    def get_proxparam1(self):
        return self.alpha

    def set_proxparam1(self, alpha):
        self.alpha = alpha

    def prox(self, f):
        u = self.O.H*(
                (self.O * (f + self.get_proxparam() * self.get_proxparam1() * self.pk) +
                    self.get_proxparam() * self.data) /
                (1 + self.tau * self.diag_sampling))
        return u

    def setP(self, pk):
        self.pk = pk.ravel()

    def getP(self):
        return self.pk