import numpy as np


class PrimalDualBase(object):

    def __init__(self):
        pass

    def update_step(self, x, x_bar, y):
        x_old = x
        # compute y^(n+1)
        y = self.update_dual(x_bar, y)
        # compute x^(n+1)
        x = self.update_primal(x, y)
        x_bar = x + self.theta*(x - x_old)

        return x, x_bar, y

    def update_primal(self,x, y):
        pass

    def update_dual(self, x_bar, y):
        pass