import numpy as np
from numpy import matlib


class Projection(object):

    def __init__(self, image_size, n_dim=2):
        self.dim = n_dim*np.prod(image_size)
        self.image_size = (image_size, n_dim)
        self.sigma = 1
        self.n_dim = n_dim

    @property
    def shape(self):
        return self.dim

    def set_proxparam(self, sigma):
        self.sigma = sigma

    def get_proxparam(self):
        return self.sigma

    def prox(self, f):
        aux = np.sqrt(np.sum(abs(np.reshape(f, (int(self.shape/self.n_dim), self.n_dim)))**2, axis=1))
        aux = matlib.repmat(np.reshape(aux, [aux.shape[0], 1]), 1, self.n_dim).ravel()  # todo
        aux[aux<1] = 1
        aux = aux.reshape(f.shape)
        u = f/aux
        return u