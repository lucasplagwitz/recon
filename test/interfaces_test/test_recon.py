import os
import unittest
import skimage
import numpy as np
from skimage import io
from scipy.sparse import diags

from pylops import Gradient

from recon.operator import MriDft
from recon.interfaces import Recon


class TestRecon(unittest.TestCase):

    def setUp(self):
        filename = os.path.join(skimage.data_dir, 'camera.png')
        self.camera = io.imread(filename)
        self.camera = self.camera[:self.camera.shape[0]//2, :self.camera.shape[1]//2]
        self.camera = self.camera/np.max(self.camera)

    def test_operator_format(self):

        with self.assertRaises(ValueError):
            id = lambda x: 1 * x
            Recon(id, domain_shape=(1,))

    def test_weight_by_zero(self):
        F = MriDft(self.camera.shape)
        k_data = F*self.camera.ravel()
        recon = Recon(F, domain_shape=self.camera.shape, reg_mode='tv', alpha=10**(-14), lam=1, tau='calc')
        u = recon.solve(k_data, tol=0.001)
        np.testing.assert_array_almost_equal(np.abs(u), self.camera)


        non_quadratic = self.camera[64:,:]
        F = MriDft(non_quadratic.shape)
        k_data = F * non_quadratic.ravel()
        recon = Recon(F, domain_shape=non_quadratic.shape, reg_mode='tv', alpha=10 ** (-14), lam=1, tau='calc')
        u = recon.solve(k_data, tol=0.001)
        np.testing.assert_array_almost_equal(np.abs(u), non_quadratic)

    def test_norm_gradient(self):
        F = MriDft((64, 64))
        grad = Gradient((64, 64) ,kind='backward')

        for _ in range(3):
            k_data = F * np.random.normal(0, 0.5, size=(4096,))
            recon = Recon(F, domain_shape=(64, 64), reg_mode='tv', alpha=0.5, lam=1, tau='calc')
            u_strong_tv = np.abs(recon.solve(k_data, tol=0.001))

            recon = Recon(F, domain_shape=(64, 64), reg_mode='tv', alpha=0.1, lam=1, tau='calc')
            u_weak_tv = np.abs(recon.solve(k_data, tol=0.001))

            self.assertGreater(np.linalg.norm(grad*u_weak_tv.ravel(), 2),
                               np.linalg.norm(grad*u_strong_tv.ravel(), 2))

    def test_undersampling(self):
        F = MriDft((64, 64))
        grad = Gradient((64, 64), kind='backward')
        sampling = diags(np.array([1, 1, 1, 1, 0, 1, 1, 1]*int(np.prod(F.image_dim)/8)))

        for _ in range(3):
            k_data = F * np.random.normal(0, 0.5, size=(4096,))
            recon = Recon(F, domain_shape=(64, 64), reg_mode='tv', alpha=0.5, lam=1, tau='calc', sampling=sampling)
            u_strong_tv = np.abs(recon.solve(k_data, tol=0.001))

            self.assertGreater(np.linalg.norm(grad * (F.inv*k_data), 2),
                               np.linalg.norm(grad * (u_strong_tv.ravel()), 2))
