import numpy as np
import pylops


from recon.terms import Projection, DatatermLinear
from recon.solver.pd_hgm import PdHgm
from recon.interfaces.base_interface import BaseInterface


class Segmentation(BaseInterface):

    def __init__(self, image_shape, classes: list, alpha: float = 0.001, tau: float = None):

        super(Segmentation, self).__init__(domain_shape=image_shape,
                                           reg_mode='tv',
                                           possible_reg_modes=['tv'],
                                           alpha=alpha,
                                           tau=tau)

        self.seg = np.zeros((np.prod(self.domain_shape), len(classes)))
        self.classes = classes

        grad = pylops.Gradient(image_shape, edge=True, dtype='float64', kind="backward")
        self.K = self.alpha * grad
        self.G = DatatermLinear()
        self.F_star = Projection(image_shape, len(self.domain_shape))
        self.solver = PdHgm(self.K, self.F_star, self.G)


    def solve(self, img, max_iter= 200, tol=10**(-6)):

        super(Segmentation, self).solve(data=img, max_iter=max_iter, tol=tol)

        self.G.set_proxdata(img.ravel())
        self.G.set_proxparam(self.tau)
        self.F_star.set_proxparam(self.tau)
        #self.solver.var['x'] = np.zeros(self.K.shape[1])
        #self.solver.var['y'] = np.zeros(self.K.shape[0])

        self.solver.max_iter = max_iter
        self.solver.tol = tol
        self.solver.solve()

        u = np.reshape(self.solver.var['x'], img.shape)
        for i in range(len(self.classes)):
            self.seg[:, i] = (u.ravel() - self.classes[i]) ** 2

        a = self.seg
        result = []
        tmp_result = np.reshape(np.argmin(a, axis=1), img.shape)
        for i, c in enumerate(self.classes):
            result.append((tmp_result == i).astype(int))
        result0 = sum([i * result[i] for i in range(len(self.classes))])
        result = np.array(result)

        return result0, result