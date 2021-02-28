import os
import skimage
from skimage import io
import numpy as np
from pylops import Gradient

from .test_datanorm_l2 import TestDatanormL2

from recon.terms import DatanormL2Bregman, IndicatorL2
from recon.solver import PdHgm


class TestDatanormL2Bregman(TestDatanormL2):

    def setUp(self) -> None:
        filename = os.path.join(skimage.data_dir, 'camera.png')
        self.camera = io.imread(filename)
        self.term = DatanormL2Bregman(image_size=self.camera.shape, data=self.camera.ravel())

    def test_iterativ_norm_convergence(self):

        G = self.term
        K = Gradient(self.camera.shape, edge=True, dtype='float64', kind='backward')
        F_star = IndicatorL2(self.camera.shape, len(self.camera.shape), prox_param=0.3)

        i = 0

        u_last = pk = np.zeros(self.camera.shape)
        pk = pk.ravel()
        while i < 10:

            self.solver = PdHgm(K, F_star, G)
            self.solver.max_iter = 350
            self.solver.tol = 0.0006
            G.pk = pk

            self.solver.solve()

            u = np.reshape(np.real(self.solver.x), self.camera.shape)
            pk = pk - (1 / G.bregman_weight_alpha) * np.real(u.ravel() - self.camera.ravel())
            i = i + 1

            self.assertLess(np.linalg.norm(u.ravel() - self.camera.ravel(), 2),
                            np.linalg.norm(u_last.ravel() - self.camera.ravel(), 2))
