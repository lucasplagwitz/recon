import numpy as np

class L2Dataterm(object):

    def __init__(self):
        self.__name__ = 'L2 Dataterm';
        self.differentiable = True;


    def primal(self, f, g):
        return 1/2 * (((f-g)**2).sum())

    def dual(self, w, g):
        return -1/2 * (w**2).sum() - (w*g).sum()

    def prox(self, z, g, tau):
        return 1 / (1 + tau) * (z + tau * g)

    def grad_f(self, f, g):
        return f - g