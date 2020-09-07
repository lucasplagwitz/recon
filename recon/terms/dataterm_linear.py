from recon.terms.base_term import BaseDataterm


class DatatermLinear(BaseDataterm):
    """
    Linear Dataterm without Operator

    Form:
        1/2 ||u - f||^2

    """

    def __init__(self):
        super(DatatermLinear, self).__init__(None)

        self.lam = 1

    def set_weight(self, lam):
        self.lam = lam

    def prox(self, f):
        """
            u =  (f + tau *f_0) / (1 + tau)
        """
        u = (f + self.get_proxparam() * self.lam * self.data) / (
            1 + self.get_proxparam() * self.lam)
        return u
