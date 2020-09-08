"""
03. Segmentation
================
This example ...
"""

###############################################################################
# TV based segmentation.

import os
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces import Segmentation

filename = os.path.join(skimage.data_dir, 'camera.png')
image = io.imread(filename, as_gray=True)

image = image/np.max(image)

classes = [0, 50/255, 120/255, 190/255, 220/255]

segmentation = Segmentation(image.shape, classes=classes, alpha=0.1, tau=3)
result, _ = segmentation.solve(image)


f = plt.figure()
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.imshow(image)
plt.title("GT")
f.add_subplot(1, 2, 2)
plt.imshow(result)
plt.title("TV based segmentation")
plt.axis('off')
plt.show(block=True)