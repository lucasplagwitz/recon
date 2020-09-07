# ToDo! handle multiple vector support correctly/automatic without times parameter

import numpy as np
from numpy import matlib

from recon.terms.base_term import BaseRegTerm


class IndicatorL2(BaseRegTerm):
    """IndicatorL2

    This function is the indicator function of the convex set P ("union of pointwise L2 balls"):
        P := {p in Y: |p| <= upper_bound}.  |p| = max_ij|p_ij| = max_ij sqrt(p1_ij**2 + p2_ij**2)

    Function d(p): {
        0       if p in P
        inf     if p not in P

    Some properties:
        - this function is the Fenchel-Legendre conjugate of L1-norm

    Imlementation based on:
        - Antonin Chambolle, Thomas Pock:
            A first-order primal-dual algorithm for convex problems with applications to imaging
        - V. Corona et al.:
            ...

    Parameter
    ---------
    image_size:

    n_dim:

    upper_bound:

    times:
        Apply multiple vectors at same time.

    """

    def __init__(self, image_size, derivate_dim=2, upper_bound: float = 1, times: int = None):
        super(IndicatorL2, self).__init__()

        self.upper_bound = upper_bound
        self.image_size = image_size
        self.derivate_dim = derivate_dim
        self.times = times

    def __call__(self, p):
        """
        Apply indicator function.
        """
        assert self._input_check(p)

        def ic(x):
            if self._infty_norm(x) <= self.upper_bound:
                return 0
            return np.inf

        if self.times == 1:
            return ic(p)
        else:
            return np.apply_along_axis(ic, 0, p)

    def prox(self, f):
        """
        Proximal Operator of indicator Function iP -> Projection.

        prox(f) = f / max(1, ||f||)

        f_row stores row-wise: In case of f^(*) as Projection and K=Grad
                                -> dual f_row[i,:] stores i-th directional derivative.
        """
        assert self._input_check(f)
        if self.times is None:
            norm_f = self._infty_abs(f)
            norm_f = matlib.repmat(norm_f, 1, self.derivate_dim).ravel() / self.upper_bound
            norm_f[norm_f < 1] = 1
        else:
            norm_list = []
            for vec in range(self.times):
                norm_f = self._infty_abs(f[:, vec])
                norm_f = matlib.repmat(norm_f, 1, self.derivate_dim).ravel() / self.upper_bound
                norm_f[norm_f < 1] = 1
                norm_list.append(norm_f)
            norm_f = np.array(norm_list).T
        return f/norm_f

    def _infty_abs(self, p):
        """
        |p_ij| = max_ij sqrt(p1_ij**2 + p2_ij**2)
        """
        p_row = np.reshape(p, (self.derivate_dim, np.prod(self.image_size)))
        norm_f = np.sqrt(np.sum(np.abs(p_row) ** 2, axis=0))
        return norm_f

    def _infty_norm(self, p):
        """
        |p| = max_ij|p_ij|
        """
        return np.max(self._infty_abs(p))

    def _input_check(self, val):
        if self.times is None or self.times == 1:
            if val.shape == np.prod(self.image_size)*self.derivate_dim:
                return True
        elif self.times > 1:
            if val.shape == (np.prod(self.image_size)*self.derivate_dim, self.times):
                return True
        print("IndicatorL2: Please ravel the input column based to shape (np.prod(self.image_size), self.times).")
        return False
