import numpy as np
from numpy import matlib

from recon.math.terms.base_term import BaseRegTerm


class Projection(BaseRegTerm):

    def __init__(self, image_size, n_dim=2):
        super(Projection, self).__init__()

        self.dim = n_dim*np.prod(image_size)
        self.image_size = (image_size, n_dim)
        self.n_dim = n_dim

    @property
    def shape(self):
        return self.dim

    def prox(self, f):
        """
        Proximal Operator of Projection.

        Prox(f) = f / max(1, ||f||)

        f_row stores row-wise: In case of f^(*) as Projection and K=Grad
                                -> dual f_row[i,:] stores i-th directional derivative.
        """

        p_row = np.reshape(f, (self.n_dim, self.shape // self.n_dim))
        norm_f = np.sqrt(np.sum(np.abs(p_row) ** 2, axis=0))
        norm_f = matlib.repmat(norm_f, 1, self.n_dim).ravel()
        norm_f[norm_f < 1] = 1
        return f/norm_f
