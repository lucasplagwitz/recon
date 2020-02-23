import numpy as np

from recon.math.primal_dual_base import PrimalDualBase

from recon.math.terms import TvIso, L2Dataterm

class L2_TV(PrimalDualBase):

    def __init__(self, sigma=1, tau=1, max_iter=100):

        super(L2_TV, self).__init__(sigma=sigma, tau=tau, max_iter=max_iter)
        self.__name__ = "Abstract Primal Dual Base class"


        self.l2_dataterm = L2Dataterm()
        self.tv = TvIso()

        self.g = None


    def update_primal(self,x, y):
        result = self.l2_dataterm.prox(x - self.tau * k_adjoint(y), self.g, self.tau)

    def update_dual(self, x_bar, y):
        result = self.tv.prox(y + self.sigma * k(x_bar))

def update_dual(self, x_bar, y):
        pass