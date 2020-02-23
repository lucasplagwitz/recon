import numpy as np
import copy

class boundary_handler_2d(object):

    def __init__(self):
        pass

    def get_value_at_idx(self, img, idx=np.array([[0, 1], [0, 1]])):
        idx0 = copy.copy(idx[:,0])
        idx1 = copy.copy(idx[:, 1])

        n, m = np.shape(img)

        if len(idx[0]) != 2:
            raise ValueError("Please use 2d index for the moment.")

        idx0[idx0 <= -1] = 0
        idx0[idx0 >= m] = n-1

        idx1[idx1 <= -1] = 0
        idx1[idx1 >= m] = m - 1

        if len(idx0[(idx0 < 0) | (idx0 > n-1)]):
            raise ValueError("Index not in rage.")

        if len(idx1[(idx1 < 0) | (idx1 > m-1)]):
            raise ValueError("Index not in rage.")

        index = tuple([idx0, idx1])
        return np.array([img[index]]).T


    def dx_value_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:, 0], idx[:,1]-1]).T)\
               - self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1]+1]).T)

    def dy_value_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:,0]-1, idx[:,1]]).T) \
               - self.get_value_at_idx(img, np.array([idx[:,0]+1, idx[:,1]]).T)

    def dxx_value_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:,0] - 1, idx[:,1]]).T) \
               - 2 * self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1]]).T) \
               + self.get_value_at_idx(img, np.array([idx[:,0] + 1, idx[:,1]]).T)

    def dyy_value_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1] - 1]).T) \
               - 2 * self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1]]).T) \
               + self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1] + 1]).T)

    def dxy_value_at_idx(self, img, idx=np.array([[0], [0]])):
        return 0.25 * (- self.get_value_at_idx(img, np.array([idx[:,0]+1, idx[:,1]-1]).T)
                       - self.get_value_at_idx(img, np.array([idx[:,0]-1, idx[:,1]+1]).T)
                       + self.get_value_at_idx(img, np.array([idx[:,0]-1, idx[:,1]+1]).T)
                       + self.get_value_at_idx(img, np.array([idx[:,0]+1, idx[:,1]-1]).T)
                       )

    def dx_plus_at_idx(self, img, idx):
        return self.get_value_at_idx(img, np.array([idx[:, 0]+1, idx[:,1]]).T)\
               - self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1]]).T)

    def dx_minus_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:, 0], idx[:,1]]).T)\
               - self.get_value_at_idx(img, np.array([idx[:,0]-1, idx[:,1]]).T)

    def dy_plus_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:, 0], idx[:,1]+1]).T)\
               - self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1]]).T)

    def dy_minus_at_idx(self, img, idx=np.array([[0], [0]])):
        return self.get_value_at_idx(img, np.array([idx[:, 0], idx[:,1]]).T)\
               - self.get_value_at_idx(img, np.array([idx[:,0], idx[:,1]-1]).T)


