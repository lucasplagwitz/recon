import numpy as np


class dimensions(object):

    def __init__(self, imagedim, components):

        if not np.shape(imagedim):
            self.image = imagedim
            self.mimage = 1
        else:
            self.image = tuple(imagedim)
            self.mimage = np.prod(np.shape(imagedim))
        self.nimage = np.prod(imagedim)
        self.components = components
        self.ntotal = self.nimage * self.components
        self.total = (self.image, components)
        self.mtotal = np.prod(self.total)