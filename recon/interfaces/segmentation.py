import numpy as np
from typing import Union


from recon.terms import IndicatorL2, DatanormL2
from recon.solver.pd_hgm import PdHgm
from recon.interfaces.base_interface import BaseInterface


class Segmentation(BaseInterface):

    def __init__(self, image_size, classes: list, lam: float = 1, alpha: float = 0.001, tau: Union[float, str] = None):

        super(Segmentation, self).__init__(domain_shape=image_size,
                                           reg_mode='tv',
                                           possible_reg_modes=['tv'],
                                           lam=lam,
                                           alpha=alpha,
                                           tau=tau)

        self.seg = np.zeros((np.prod(self.domain_shape), len(classes)))
        self.classes = classes
        self.G = DatanormL2(image_size=image_size, prox_param=self.tau, lam=self.lam)

    def solve(self, img, max_iter=2000, tol=1e-4):

        data = np.zeros((np.prod(self.domain_shape), len(self.classes)))
        for i in range(len(self.classes)):
            data[:, i] = (img.ravel() - self.classes[i]) ** 2

        super(Segmentation, self).solve(data=data, max_iter=max_iter, tol=tol)
        self.F_star = IndicatorL2(self.domain_shape,
                                  len(self.domain_shape),
                                  times=len(self.classes),
                                  prox_param=self.tau,
                                  upper_bound=self.alpha)
        self.solver = PdHgm(self.K, self.F_star, self.G)

        self.G.data = data
        self.solver.x = np.zeros((self.K.shape[1], len(self.classes)))
        self.solver.y = np.zeros((self.K.shape[0], len(self.classes)))

        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        u = np.reshape(self.solver.x, tuple(list(img.shape) + [len(self.classes)]), order='C')

        a = u
        result = []
        tmp_result = np.argmin(a, axis=len(img.shape))
        for i, c in enumerate(self.classes):
            result.append((tmp_result == i).astype(int))
        result0 = sum([i * result[i] for i in range(len(self.classes))])
        result = np.array(result)

        return result0, result
