import numpy as np

class TvIso(object):

    def __init__(self):
        self.__name__ = 'Isotropic TV Regularizer'
        self.differentiable = False

    def primal(self, grad_f):
        #shape = np.shape(grad_f)
        return np.sqrt((grad_f**2).sum())

    def dual(self):
        return 0

    def prox(self, y, alpha):
        shape = np.shape(y)

        if len(shape) < 2 or len(shape) > 3:
            raise NotImplementedError("Not implemented for your dimension.")
        else:  # {2, 3}-dim Image
            return (alpha*y) / max([alpha, np.sqrt((y**2).sum())])