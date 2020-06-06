import numpy as np

from recon.math.terms.base_term import BaseTerm

class Dataterm(BaseTerm):
    """
        This class is the basic form of the L2-Dataterm with undersampling matrics.

        Form:
        u(f) = argmin_x 1/2 * ||(S*F)*x - f||^2

        """

    def __init__(self, O, S):

        super(Dataterm, self).__init__(O)

        self.proxdata = np.zeros(S.shape[0])
        self.S = S
        self.diagS = (S.T*S).diagonal()

    def modify_data(self):
        self.data = (self.S.T)*self.proxdata

    def set_proxdata(self, proxdata):
        self.proxdata = proxdata.T.ravel()
        self.modify_data()

    def prox(self, f):
        """
        u = F_star * (( F*f + tau * S_star*f_0) / (1 + tau * S_star*S)
        """
        if self.pylops:
            F_star = self.F.H
        else:
            F_star = self.F.T

        u = F_star*(
                    (self.F*f + self.get_proxparam() * (self.data)) /
                    (1 + self.get_proxparam() * self.diagS)
                )
        return u