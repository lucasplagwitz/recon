"""
07. Segmentation
================
This example shows how to use the interface for class-based segmentation
of 2D images. First, depending on the size of the weighting alpha,
a piecewise constant image is generated before the assignment to certain classes is done.
"""

###############################################################################
# TV based segmentation
import skimage.data as skd
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces import Segmentation

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

gt = rgb2gray(skd.coffee())[:,80:481]
gt = gt/np.max(gt)
gt = gt/np.max(gt)

classes = [0, 50/255, 120/255, 190/255, 220/255]

segmentation = Segmentation(gt.shape, classes=classes, lam=5, tau='calc')
result, _ = segmentation.solve(gt, max_iter=4000)

f = plt.figure(figsize=(6, 3))
f.add_subplot(1, 2, 1)
plt.axis('off')
plt.imshow(gt)
plt.title("GT")
f.add_subplot(1, 2, 2)
plt.imshow(result)
plt.title("TV-based segmentation")
plt.axis('off')
plt.show(block=False)
