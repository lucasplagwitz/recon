import numpy as np

import imageio
import matplotlib.pyplot as plt

from recon.segmentation.chan_vese import chan_vese

image_path = "./../../examples/data/gt.png"
image = imageio.imread(image_path, as_gray=True)
image = image/np.max(image)*255
chan_vese(image)