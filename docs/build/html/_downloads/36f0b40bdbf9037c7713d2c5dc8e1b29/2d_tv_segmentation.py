"""
03. Segmentation
================
This example shows how to use the interface for class-based segmentation
of 2D images. First, depending on the size of the weighting alpha,
a piecewise constant image is generated before the assignment to certain classes is done.
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

segmentation = Segmentation(image.shape, classes=classes, alpha=0.01, tau='calc')
result, _ = segmentation.solve(image)


f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.imshow(image)
plt.title("GT")
f.add_subplot(1, 2, 2)
plt.imshow(result)
plt.title("TV based segmentation")
plt.axis('off')
plt.show(block=False)