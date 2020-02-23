import numpy as np

import imageio
import matplotlib.pyplot as plt

from recon.segmentation.chan_vese import c1

image_path = "./../../examples/gt.png"
image = imageio.imread(image_path, as_gray=True)
image = image/np.max(image)*255
c1(image)