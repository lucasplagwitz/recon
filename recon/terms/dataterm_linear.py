from recon.terms.base_term import BaseDataterm


class DatatermLinear(BaseDataterm):
    """
    Linear Dataterm without Operator

    Form:
        1/2 ||u - f||^2

    """

    def __init__(self):
        super(DatatermLinear, self).__init__(None)


    def prox(self, f):
        """
            u =  (f + tau *f_0) / (1 + tau)
        """
        u = (f + self.get_proxparam() * self.data) / (
            1 + self.get_proxparam())
        return u
