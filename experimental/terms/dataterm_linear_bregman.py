from recon.terms import BaseDataterm


class DatatermLinearBregman(BaseDataterm):

    def __init__(self):
        super(DatatermLinearBregman, self).__init__(None)
        self.p = 0
        self.alpha = 1

    def get_proxparam1(self):
        return self.alpha

    def set_proxparam1(self, alpha):
        self.alpha = alpha

    def setP(self, p):
        self.p = p

    def getP(self):
        return self.p

    def prox(self, f):
        u = ((f+ self.get_proxparam() * self.get_proxparam1() * self.p)) / (1 + self.tau)
        return u