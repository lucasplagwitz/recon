import numpy as np
import pylops

class Dataterm(object):
    """
        This class is the basic form of the L2-Dataterm.

        Form:
        u(f) = argmin_x 1/2 * ||Ax - f||^2     A=S*F
        u_0(f) = 1/2 * ||Ax - f||^2     A=S*F
        u'(f) = AT * (Ax - f)
        prox_{tau * u(f)}(x) = FT * ST * (SF * x - f) = FT * (STS * F * x - ST * f) =(1)=

        (*) prox_{tau * f}(f_bar) = (I + tau*df)^{-1}(f_bar)
        prox_{tau * u_0}(f) = (I + ATAf - ATx)^{-1}(f) = (I + tau * (FT * (STS * F * f - ST * x))^{-1}(f)

        IF S = I
        = (I + tau * (FT*F * f - FT*x))^{-1}(f)





        = (I + tau* FT*F*f - tau*FT*x)^{-1}(f)





        u = 1/(1+tau) * FT*F*f + tau*Ft*x

        u = FT*F*f/(1+tau) + tau/(1+tau)(Ft*x)

        u = FT (F*f/(1+tau*diagS) + tau*Ft*(x/(1+tau*diagS))
        (1)

        """

    def __init__(self, S, F):
        """

        :param S: sampling Matrix for
        :param F: Operator
        """
        self.dim = F.shape[1]
        self.tau = 1
        if S is not None:
            self.proxdata = np.zeros(S.shape[0])
            self.S = S
            self.diagS = (S.T*S).diagonal()
        else:
            self.diagS = 1
            self.S = None
        self.F = F

        if isinstance(F, pylops.LinearOperator):
            self.pylops = True

    def modify_data(self):
        if self.S is not None:
            self.data = (self.S.T)*self.proxdata
        else:
            self.data = self.proxdata

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def set_proxdata(self, proxdata):
        self.proxdata = proxdata.T.ravel()
        self.modify_data()

    def prox(self, f):
        """
        u = F_star * (( F*f + tau * S_star*f_0) / (1 + tau * S_star*S)
        u = f + phi * A_star * f_0 /
        :param f:
        :return:
        """
        if self.pylops:
            F_star = self.F.H
        else:
            F_star = self.F.T
        u = F_star*((self.F*f + self.get_proxparam() * (self.data)) / (
            1 + self.get_proxparam() * self.diagS))
        return u