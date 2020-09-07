from recon.terms.base_term import BaseDataterm

from pylops import Diagonal

class DatatermLinearConstrained(BaseDataterm):
    """
    Linear Dataterm without Operator

    Form:
        1/2 lam*||u - f||^2

    """

    def __init__(self, lam=1):
        super(DatatermLinearConstrained, self).__init__(None)
        self.lam = Diagonal(lam)
        if not isinstance(lam, int):
            self.diag_lam = (self.lam.T * self.lam)
        else:
            self.diag_lam = lam


    def prox(self, f):
        """
            u =  (f + tau *f_0) / (1 + tau)
        """
        u = (f + self.lam*self.get_proxparam() * self.data) / (
            1 + self.diag_lam*self.get_proxparam())
        return u
