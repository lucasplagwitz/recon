import numpy as np

import imageio
import matplotlib.pyplot as plt

from recon.segmentation.chan_vese_multi_dim import chan_vese

image_path = "./data/gt.png"
image = imageio.imread(image_path, as_gray=True)
image = image/np.max(image)*255
nu = 0.000001
classes = []
n_seg = 4

result = chan_vese(image, n_segments=n_seg+1, classes=classes, nu=nu)


plt.Figure()
plt.imshow(result)
plt.xlabel('Chan-Vese segmentation')
plt.axis('off')
plt.savefig("./data/output/chanvese_multiclass_segmentation.png")
plt.close()