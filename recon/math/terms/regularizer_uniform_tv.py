# MATLAB Code from V.Corona
# https://github.com/veronicacorona/JointReconstructionSegmentation/blob/master/JointRecSegm/convex/regularizer_uniform_tv.m

# Author: Jan Lellmann, j.lellmann@damtp.cam.ac.uk
# python implementation: l.plagwitz@uni-muenster.de
# creates a forward-difference p-norm regularizer with Neumann boundary
# conditions, e.g. lambda * TV(A (*) u), where A (*) u means A is applied
# to u(x) in each point uniformly. If A is not supplied, it is assumed to
# be the identity. A does not necessarily have to be quadratic.

import numpy as np
from scipy import sparse
from recon.math.operator.first_derivative import FirstDerivative
#from scipy.sparse import linalg

def regularizer_uniform_tv(dim,
                           lam,
                           p,
                           schemes,
                           boundaries,
                           A: np.ndarray = np.eye(1)):

    if A == np.eye(1):
        A = np.eye(dim.components)


    components = A.shape[0]


    righthand = sparse.kron(sparse.csr_matrix(A), sparse.eye(dim.nimage))
    righthandbound = np.linalg.norm(A) # here only frobenius not 2 norm -> maybe cast?
                                           # this should be computable, as A is usually small
    #nabla, nablabound = diffopn(dim.image, components, schemes, boundaries)

    if type(dim.image) == tuple:
        n = dim.image[1]
    else:
        n = dim.image

    nabla = FirstDerivative(righthand.shape[1], boundaries=boundaries)
    op = nabla * righthand
    opnormbound = 2.828


    # construct dual norm by the rule 1/p + 1/dualp = 1
    if p == 1:
        raise ValueError('p = 1 has no well-defined meaning; consider using regularizer_uniformanisotropic_tv')
        #dualp = np.inf
        #dualradius = np.sqrt(l1, 1))
    elif p == 2:
        dualp = 2
        dualradius = np.sqrt(dim.nimage)
    else:
        raise NotImplementedError('p must be 1 or 2')

    result = {}

    # primal methods
    result['operator'] = lambda x: (op * x.ravel())
    result['evaluate'] = lambda x: regularizer_uniform_tv_evaluate(lam,
                                                                np.reshape(op * x.ravel(),
                                                                                [dim.nimage(dim.mimage * components)]),
                                                                p)
    result['b'] = np.zeros( (op.shape[0], 1) )

    # dual methods
    result['adjoint_operator'] = lambda y: op.T * y.ravel()

    result['dual_constraints'] = constraints(
                                    lambda y: np.reshape(project_pnorm_uniform(
                                        np.reshape(y, [dim.nimage(dim.mimage * components)]), lam , dualp), (op.shape[0], 1)),
                                           # projector

                                    lam * np.zeros( (op.shape[0], 1) ),  # center
                                    lam * dualradius # radius % FIXME pessimistic estimate of the
                                                      # RADIUS in 2 - norm(here: for dualp = inf)
                                    )

    result['dual_constraints'].liftable = True
    result['dual_constraints'].lift = lambda x: x
    result['dual_constraints'].unlift = lambda x: x
    result['dual_constraints'].lifted_project = result['dual_constraints'].project
    result['dual_constraints'].lift_factor = 1

    # general information
    result['lengths'] = op.shape

    # set operator norm
    result['operator_norm'] = opnormbound

    # special functions t = (u, f)
    #result.dual_from_primal = lambda t: (dfp_regularizer_uniform_tv(t[0], t[1], result, lam, dim))

    if (p == 2):
        # t = (M, N)
        result['discretization_distance'] = lambda t: (lam * np.sqrt(sum(((A * ((t[0] - t[1]).T)).T)**2, 2)))

    #result.operator_backward_step = backward_step_gradient_dct(dim, schemes, boundaries, A); % DEBUG
    #if (isempty(result.operator_backward_step))
    #    result = rmfield(result, 'operator_backward_step');
    #end


    #result.operator_as_matrix = lambda _ : op


    print("test")



def regularizer_uniform_tv_evaluate():
    pass



class constraints(object):

    def __init__(self, projector, center, radius):

        self.project = projector
        #if (exist('center','var'))
        self.center = center

        #if (exist('radius','var'))
        self.radius = radius


def project_pnorm_uniform(vectors, length, p):
    """
    EUCLIDEAN projection of all rows (!) in input to unit sphere of radius 1en in p-norm
    currently implemented for p=2, p=inf
    WARNING: for p other than 2,inf, projection is NOT trivial!
    :param vectors:
    :param len:
    :param p:
    :return:
    """
    if (p == 2):
        norms = np.sqrt(sum(vectors**2, 2)) # row vector
        with norms:
            factors = 1./ np.max(np.ones(length), norms / length)
        result = sparse.spdiags(factors, diags=0, m=length, n=length) * vectors # SLOW
    elif p == np.inf:
        norms = abs(vectors)
        result = vectors/np.max(np.ones(length), norms/length) # SLOW
    else:
        raise ValueError('p must be 2 or +inf')

    return result