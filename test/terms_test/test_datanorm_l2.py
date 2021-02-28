import os
import unittest
import skimage
import numpy as np
from skimage import io

from recon.operator import MriDft

from recon.terms import DatanormL2


class TestDatanormL2(unittest.TestCase):

    def setUp(self) -> None:
        filename = os.path.join(skimage.data_dir, 'camera.png')
        self.camera = io.imread(filename)[50:250, 50:250]
        self.term = DatanormL2(image_size=self.camera.shape, data=self.camera.ravel())

    def test_callable(self):
        self.assertAlmostEqual(self.term(self.camera.ravel()), 0)

        F = MriDft(self.camera.shape)
        data = F * self.camera.ravel()
        indicator = DatanormL2(operator=F, image_size=self.camera.shape, data=data)
        self.assertAlmostEqual(indicator(self.camera.ravel()), 0)

    def test_prox_linear(self):
        prox_camera = np.zeros(shape=self.camera.ravel().shape)

        # apply prox => decrease indicator => converge fast to zero
        for i in range(50):
            prev_camera = prox_camera
            prox_camera = self.term.prox(prox_camera)
            self.assertGreaterEqual(self.term(prev_camera), self.term(prox_camera))

        self.assertAlmostEqual(self.term(prox_camera), 0, 4)

    def test_prox_operator(self):
        F = MriDft(self.camera.shape)
        raveled_camera = self.camera.ravel()
        data = F*raveled_camera
        self.term = DatanormL2(operator=F,
                               image_size=self.camera.shape,
                               data=data,
                               prox_param=0.1)
        prox_camera = np.zeros(shape=raveled_camera.shape)

        # apply prox => decrease indicator => converge fast to zero
        for i in range(150):
            prev_camera = prox_camera
            prox_camera = self.term.prox(prox_camera)
            self.assertGreaterEqual(self.term(prev_camera), self.term(prox_camera))

        np.testing.assert_almost_equal(raveled_camera, prox_camera, 3)

    #def test_unraveld(self):
    #    with self.assertRaises(ValueError):
    #        indicator = DatanormL2(image_size=self.camera.shape, data=self.camera)
    #        prox_camera = np.zeros(shape=self.camera.shape)
    #        _ = indicator.prox(prox_camera)