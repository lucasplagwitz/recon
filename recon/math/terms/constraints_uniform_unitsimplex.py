# Author: Jan Lellmann, j.lellmann@damtp.cam.ac.uk
# If one argument is given, projects each vector separately onto the unit
# simplex

# (obsolete) If basis is supplied, it must be a (k x dims.components)
# matrix; in this case each vector is projected on the intersection of the
# unit simplex with the orthogonal complement of the kernel of basis.
# Set basis = [] to disable.

# mask and border can be used to specify "ghost" cells that are fixed.
# mask ist an index vector of size dims.ntotal x 1; border is a
# numerical t x 1 vector with t the number of "true" elements in mask.
# At each iteration, these cells are reset via u(mask) = border(:).
# If mask contains an entry for one component at a pixel, it must contain
# entries for all other components at that pixel as well; otherwise the
# behavior is undefined!

import numpy as np

class ConstraintsUniformUnitsimplex(object):

    def __init__(self, dims, basis=None, mask = {}, border = {}):

        spproj = self.projector_uniform_unitsimplex(dims)
        self.dims = dims

    def prox(self, u):
        result = np.reshape(self.project_unitsimplex(np.reshape(u, (self.dims.nimage, self.dims.components)),
                                                (self.dims.ntotal, 1)))
        return result

    @staticmethod
    def projector_uniform_unitsimplex(dims):
        result = lambda u: np.reshape(
                            ConstraintsUniformUnitsimplex.project_unitsimplex(np.reshape(u, (dims.nimage, dims.components)),
                                                (dims.ntotal, 1)))

        return result

    @staticmethod
    def project_unitsimplex(c):
        """
        #PROJECT_UNITSIMPLEX Projects ROWS of c onto the unit simplex
        #   i.e. (euclidean) projection onto the set sum(flatten(x)) = 1, x >= 0.
        #   Algorithm converges in at most numel(c) steps, O(n^2) total (could be
        #   lower)
        #   Source: "C.Michelot: A Finite Algorithm for Finding the Projection of a
        #            Point onto the Canonical Simplex of R^n", JOTA Vol.50, No.1,
        #            July 1986
        :param c:
        :return:
        """

        c = c.T # algorithm is formulated for columns

        zeroset = np.zeros(c.shape)
        n = c.shape[0]
        if len(c.shape) > 1:
            m = c.shape[1]
        else:
            m = 1
        ni = n * np.ones((1, m))

        x = c

        finished = False
        while not finished:
            s = np.sum(x, 0)
            #x = (x - np.matlib.repmat( (s-1)/ni, c.shape[0], 1)) * (1 - zeroset)
            x = (x - (s - 1) / ni) * (1 - zeroset)
            conflicts = np.where(x < 0)

            if not any(conflicts[0]):
                finished = True
            else:
                zeroset[conflicts] = 1
                ni = n - np.sum(zeroset, 0)

            x[conflicts] = 0

        return x.T

    def project_mask(u, mask, border):
        u[mask] = border
        return u


