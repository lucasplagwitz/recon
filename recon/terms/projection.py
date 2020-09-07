import numpy as np
from numpy import matlib

from recon.terms.base_term import BaseRegTerm

# ToDo! handle multiple vector support correctly


class Projection(BaseRegTerm):

    def __init__(self, image_size, n_dim=2, upper_bound=1, times=1):
        super(Projection, self).__init__()

        self.upper_bound = upper_bound
        self.image_size = image_size
        self.n_dim = n_dim
        self.times = times  # allow multiple vectors add sample time

    def prox(self, f):
        """
        Proximal Operator of indicator Function iC -> Projection.

        Prox(f) = f / max(1, ||f||)

        f_row stores row-wise: In case of f^(*) as Projection and K=Grad
                                -> dual f_row[i,:] stores i-th directional derivative.
        """
        if self.times == 1:
            p_row = np.reshape(f, (self.n_dim, np.prod(self.image_size)))
            norm_f = np.sqrt(np.sum(np.abs(p_row) ** 2, axis=0))
            norm_f = matlib.repmat(norm_f, 1, self.n_dim).ravel()/self.upper_bound
            norm_f[norm_f < 1] = 1
        else:
            norm_list = []
            for vec in range(self.times):
                p_row = np.reshape(f[:, vec], (self.n_dim, np.prod(self.image_size)))
                norm_f = np.sqrt(np.sum(np.abs(p_row) ** 2, axis=0))
                norm_f = matlib.repmat(norm_f, 1, self.n_dim).ravel()/self.upper_bound
                norm_f[norm_f < 1] = 1
                norm_list.append(norm_f)
            norm_f = np.array(norm_list).T
        return f/norm_f
