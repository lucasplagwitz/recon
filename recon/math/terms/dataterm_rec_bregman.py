import pylops

from recon.math.terms.base_term import BaseTerm

class DatatermRecBregman(BaseTerm):

    def __init__(self, O):
        super(DatatermRecBregman, self).__init__(O)

        self.alpha = 1
        if isinstance(self.O, pylops.LinearOperator):
            self.pylops = True

    def get_proxparam1(self):
        return self.alpha

    def set_proxparam1(self, alpha):
        self.alpha = alpha

    def set_proxdata(self, proxdata):
        self.data = proxdata.T.ravel()

    def prox(self, f):
        if self.pylops:
            O_star = self.O.H
        else:
            O_star = self.O.T
        u = O_star*((self.O * (f+ self.get_proxparam() * self.get_proxparam1() * self.pk) +
                       self.get_proxparam() * self.data) / (
                        1 + self.tau))
        return u

    def setP(self, pk):
        self.pk = pk.T.ravel()

    def getP(self):
        return self.pk