#
# Author: Jan Lellmann, j.lellmann @ damtp.cam.ac.uk
#

# creates 1-dimensional sparse difference operator matrix for
# vectors with length n boundaries can be a string or a cell array of
# 2 strings for left and
# right boundary conditions
# normbnd is optional returns an upper bound for the operator's norm if
# available( and +inf if not).
# NOTE: A grid spacing of _2_ is assumed!

"""
# Gradient  operator
ex = np.ones((gt.shape[1],1))
ey = np.ones((1, gt.shape[0]))
dx = sparse.diags([1, -1], [0, 1], shape=(gt.shape[1], gt.shape[1])).tocsr()
dx[gt.shape[1]-1, :] = 0
dy = sparse.diags([-1, 1], [0, 1], shape=(gt.shape[0], gt.shape[0])).tocsr()
dy[gt.shape[0]-1, :] = 0

grad = sparse.vstack((sparse.kron(dx, sparse.eye(gt.shape[0]).tocsr()),
                      sparse.kron(sparse.eye(gt.shape[1]).tocsr(), dy)))
"""


import numpy as np
from scipy import sparse

from recon.math.operator.fcthdl_operator import FcthdlOperator

class FirstDerivative(FcthdlOperator):

    def __init__(self, n: int, boundaries: str):
        """
        Limited implentation of diffop in Matlab version
        :param n:
        :param boundaries:
        """
        result = []
        normbound = np.inf

        self.boundaries = boundaries

        if n < 1:
            raise ValueError('n must be positive')

        def fwfcthdl(righthand, righthandbound = None):
            """
            Only Neumann at the moment!
            TODO improve estimate for neumann & dirichlet conditions (can
            probably be computed explicitly)
            TODO look up proof for this (experimentally verified using svd(...); for periodic conditions:
            2.0 is exactly reached.
            :param righthand:
            :param righthandbound:
            :return:
            """
            if self.boundaries == 'neumann':
                result = sparse.diags([np.array([-1]*(n-1) + [0]), np.ones(n-1)], [0, 1])
                normbound = 2
            else:
                raise NotImplementedError("Other boundaries than neumann not implemented yet!")

            return result * righthand

        def bwfcthdl(lefthand):
            if self.boundaries == 'neumann':
                #result =  sparse.diags([np.ones(n-1), -np.array([1]*(n-1) + [0])], [-1, 0]) #not sure
                result = sparse.diags([-np.ones(n-i) for i in range(n)], list(range(n)))
                normbound = 2
            else:
                raise NotImplementedError("Other boundaries than neumann not implemented yet!")

            return lefthand * result

        super(FirstDerivative, self).__init__(n, n, fwfcthdl, bwfcthdl)
