import os
import unittest
import skimage
import numpy as np
from skimage import io

from pylops import Gradient

from recon.terms import IndicatorL2


class TestIndicatorL2(unittest.TestCase):

    def setUp(self) -> None:
        filename = os.path.join(skimage.data_dir, 'camera.png')
        self.camera = io.imread(filename)


    def test_callable(self):
        # base test for derivate_dim == 1
        derivate_dim = 1
        image_size = (64, 64)
        indicator = IndicatorL2(image_size=image_size, derivate_dim=1, upper_bound=1)

        # should 0 cause random.uniform in (0, 1)
        for _ in range(20):
            self.assertEqual(indicator(np.random.uniform(size=image_size).ravel()), 0)


        # should inf cause upper |p| > 0
        indicator = IndicatorL2(image_size=image_size, derivate_dim=derivate_dim, upper_bound=-1)
        for _ in range(20):
            self.assertEqual(indicator(np.random.uniform(size=image_size).ravel()), np.inf)

        # base test for derivate_dim == 2
        grad = Gradient(dims=self.camera.shape, edge=True, kind='backward')
        g_camera = grad*self.camera.ravel()
        indicator = IndicatorL2(image_size=self.camera.shape, derivate_dim=2, upper_bound=1)
        self.assertEqual(indicator(g_camera), np.inf)

        # times > 1
        indicator = IndicatorL2(image_size=image_size, derivate_dim=1, upper_bound=1, times=20)
        np.testing.assert_array_equal(indicator(np.random.uniform(size=(np.prod(image_size), 20))),
                                      np.zeros(20))

        indicator = IndicatorL2(image_size=image_size, derivate_dim=1, upper_bound=-0.1, times=20)
        np.testing.assert_array_equal(indicator(np.random.uniform(size=(np.prod(image_size), 20))),
                                      np.array([np.inf]*20))

    def test_prox(self):
        grad = Gradient(dims=self.camera.shape, edge=True, kind='backward')
        g_camera = grad * self.camera.ravel()
        indicator = IndicatorL2(image_size=self.camera.shape, derivate_dim=2, upper_bound=1)
        cur_camera = prev_camera = g_camera
        start_camera = 0
        # apply prox => decrease indicator => converge fast to zero
        for i in range(4):
            cur_camera = indicator.prox(cur_camera)
            self.assertTrue(indicator(prev_camera) >= indicator(cur_camera))
            prev_camera = cur_camera
            if i == 0:
                start_camera = cur_camera
        self.assertEqual(indicator(cur_camera), 0)

        indicator = IndicatorL2(image_size=self.camera.shape, derivate_dim=2, upper_bound=1, times=10)
        mult_camera = np.array([g_camera]*10).T
        prox_results = indicator.prox(mult_camera)
        for i in range(10):
            np.testing.assert_array_equal(prox_results[:,i], start_camera)
