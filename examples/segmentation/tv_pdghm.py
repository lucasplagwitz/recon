import numpy as np

import imageio
import matplotlib.pyplot as plt

from recon.segmentation.tv_pdghm import multi_class_segmentation

image_path = "./../../examples/data/gt.png"
image = imageio.imread(image_path, as_gray=True)
image = image/np.max(image)


classes = [0, 50/255, 100/255, 160/255, 210/255]


result, _ = multi_class_segmentation(image, classes=classes, beta=0.001)


plt.Figure()
plt.imshow(result.T)
plt.xlabel('TV based segmentation')
plt.axis('off')
plt.savefig("./../data/segmentation/output/tv_segmentation.png")
plt.close()