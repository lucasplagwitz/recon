import skimage
import os
from skimage import io
import unittest
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

from recon.operator.ct_radon import CtRt


class TestCtRt(unittest.TestCase):

    def test_forward(self):
        filename = os.path.join(skimage.data_dir, 'camera.png')
        camera = io.imread(filename)
        camera = camera[20:250, 20:250]
        camera = camera / np.max(camera)
        theta = np.linspace(0., 180., 180, endpoint=False)
        R = CtRt(camera.shape, theta=theta)
        sol = np.reshape(R.inv * R * camera.ravel(), camera.shape)

        sino = radon(camera, R.theta, circle=False)
        sol2 = iradon(sino, R.theta, circle=False)

        np.testing.assert_array_almost_equal(R * camera.ravel(), sino.ravel())
        np.testing.assert_array_almost_equal(sol2, sol)


    #def test_id(self):
        # obviously false
    #    filename = os.path.join(skimage.data_dir, 'camera.png')
    #    camera = io.imread(filename)
    #    camera = camera[20:250, 20:250]
    #    camera = camera / np.max(camera)
    #    theta = np.linspace(0., 180., 180, endpoint=False)
    #    R = CtRt(camera.shape, theta=theta)
    #    sol = np.reshape(R.inv * R * camera.ravel(), camera.shape)
    #    #sol = sol / np.max(sol)
    #    plt.imshow(sol)
    #    plt.show()
    #    plt.imshow(camera)
    #    plt.show()
    #    print(np.mean(np.abs(sol-camera)))
    #    np.testing.assert_array_almost_equal(sol, camera)