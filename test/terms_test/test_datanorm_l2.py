import os
import unittest
import skimage
import numpy as np
from skimage import io

from pylops.signalprocessing import FFT2D

from recon.terms import DatanormL2


class TestDatanormL2(unittest.TestCase):

    def setUp(self) -> None:
        filename = os.path.join(skimage.data_dir, 'camera.png')
        self.camera = io.imread(filename)
        self.term = DatanormL2(image_size=self.camera.shape, data=self.camera.ravel())

    def test_callable(self):
        self.assertAlmostEqual(self.term(self.camera.ravel()), 0)

        F = FFT2D(self.camera.shape)
        sino = F * self.camera.ravel()
        indicator = DatanormL2(operator=F, image_size=self.camera.shape, data=sino)
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
        F = FFT2D(self.camera.shape)
        raveled_camera = self.camera.ravel()
        sino = F*raveled_camera
        self.term.operator = F
        self.term.data = sino
        prox_camera = np.zeros(shape=raveled_camera.shape)

        # apply prox => decrease indicator => converge fast to zero
        for i in range(30):
            prev_camera = prox_camera
            prox_camera = self.term.prox(prox_camera)
            self.assertGreaterEqual(self.term(prev_camera), self.term(prox_camera))

        np.testing.assert_almost_equal(raveled_camera, prox_camera, 4)

    def test_unraveld(self):
        with self.assertRaises(ValueError):
            indicator = DatanormL2(image_size=self.camera.shape, data=self.camera)
            prox_camera = np.zeros(shape=self.camera.shape)
            _ = indicator.prox(prox_camera)