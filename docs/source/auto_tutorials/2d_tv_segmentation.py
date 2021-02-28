"""
03. Segmentation
================
This example ...
"""

###############################################################################
# We import ....

import os
import skimage
from skimage import io
import numpy as np
import imageio
import matplotlib.pyplot as plt

from recon.interfaces import Segmentation

filename = os.path.join(skimage.data_dir, 'camera.png')
image = io.imread(filename, as_gray=True)

image = image/np.max(image)

classes = [0, 50/255, 100/255, 160/255, 210/255]



segmentation = Segmentation(image.shape, classes=classes, alpha=0.1, tau=3)
result, _ = segmentation.solve(image)

plt.Figure()
plt.imshow(result)
plt.xlabel('TV based segmentation')
plt.axis('off')
#plt.savefig("./data/output/2d_tv_segmentation.png", bbox_inches = 'tight', pad_inches=0)
#plt.close()