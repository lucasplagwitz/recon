from recon.math.terms.base_term import BaseTerm

class Dataterm(BaseTerm):
    """
        This class is the basic form of the L2-Dataterm.

        Form:
        u(f) = argmin_x 1/2 * ||Ax - f||^2
    """

    def __init__(self, O):
        super(Dataterm, self).__init__(O)


    def set_proxdata(self, proxdata):
        self.data = proxdata.ravel() #.T.ravel()

    def prox(self, f):
        """
        Proximal Operator

        u = F_star * (( F*f + tau * S_star*f_0) / (1 + tau * S_star*S)
        """
        if self.pylops:
            O_star = self.O.H
        else:
            O_star = self.O.T

        u = O_star*(
                (self.O*f + self.get_proxparam() * self.data) /
                (1 + self.get_proxparam())
                )

        return u