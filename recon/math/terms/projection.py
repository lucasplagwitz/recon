import numpy as np
from numpy import matlib


class Projection(object):

    def __init__(self, image_size):
        self.dim = 2*np.prod(image_size)
        self.image_size = (image_size, 2)
        self.sigma = 1

    @property
    def shape(self):
        return self.dim

    def set_proxparam(self, sigma):
        self.sigma = sigma

    def get_proxparam(self):
        return self.sigma

    def prox(self, f):
        aux = np.sqrt(np.sum(abs(np.reshape(f, (int(self.shape/2), 2)))**2, axis=1))
        aux = matlib.repmat(np.reshape(aux, [aux.shape[0], 1]),1,2).ravel()
        aux[aux<1] = 1
        u = f/aux
        return u