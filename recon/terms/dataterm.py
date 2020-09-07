from recon.terms.base_term import BaseDataterm


class Dataterm(BaseDataterm):
    """
        This class is the basic form of the L2-Dataterm.

        Form:
        u(f) = argmin_x 1/2 * ||Ax - f||^2
    """

    def __init__(self, O, sampling=None):
        super(Dataterm, self).__init__(O, sampling=sampling)
        self.lam = 1
        self.faktor = 1

    def set_weight(self, lam):
        self.lam = lam

    def prox(self, f):
        """
        Proximal Operator

        u = F_star * (( F*f + tau * S_star*f_0) / (1 + tau*diag_sampling)
        """
        u = self.O.H*(
                (self.O*f + self.get_proxparam() * self.lam * self.data) /
                (1 + self.get_proxparam() * self.lam * self.diag_sampling)
            )

        return self.faktor * u