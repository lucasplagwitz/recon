import skimage
import os
from skimage import io
import unittest
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

from recon.operator.mri_dft import MriDft


class TestMriDft(unittest.TestCase):

    def test_id(self):
        filename = os.path.join(skimage.data_dir, 'camera.png')
        camera = io.imread(filename)
        camera = camera / np.max(camera)
        F = MriDft(camera.shape)
        sol = np.reshape(F.inv * F * camera.ravel(), camera.shape)
        print(np.mean(np.abs(sol-camera)))
        np.testing.assert_array_almost_equal(sol, camera)