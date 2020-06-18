import numpy as np

import imageio
import matplotlib.pyplot as plt

from experimental.segmentation.chan_vese import chan_vese

image_path = "./data/gt.png"
image = imageio.imread(image_path, as_gray=True)
image = image/np.max(image)*255

result = chan_vese(image)

plt.Figure()
plt.imshow(result)
plt.xlabel('Chan-Vese segmentation')
plt.axis('off')
plt.savefig("./data/output/chanvese_singleclass_segmentation.png")
plt.close()