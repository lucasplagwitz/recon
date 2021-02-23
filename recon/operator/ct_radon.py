import numpy as np
import pylops
import odl

from recon.operator.fcthdl_operator import FcthdlOperator


class CtRt(FcthdlOperator):
    """
    MRI FourierTransform Operator.

    fwfcthdl: RT
    bwfcthdl: IRT
    """

    def __init__(self, domain_dim, center=np.array([0]), theta=None, norm=1, limit=np.pi):

        self.domain_dim = domain_dim  # attention: self doppelt...
        self.center = center

        self.norm = norm

        if not any(self.center):
            if len(self.domain_dim) == 2 and self.domain_dim[1] == 1:
                self.center = int((self.domain_dim[0] + 1) / 2)
            else:
                self.center = np.array([x//2 for x in list(domain_dim)])

        # ToDO: input check

        if len(center) == 1:
            self.onevec = 1
        else:
            self.onevec = np.ones(len(self.domain_dim))

        roll_val = self.center - self.onevec

        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.linspace(0., 180., 180, endpoint=False) # max(self.domain_dim)

        ######## odl -------
        reco_space = odl.uniform_discr(
            min_pt=[-20, -20], max_pt=[20, 20], shape=[domain_dim[0], domain_dim[1]], dtype='float32')

        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(0, limit, len(self.theta)) #-0.5
        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        detector_partition = odl.uniform_partition(-30, 30, np.max(domain_dim))
        # Make a parallel beam geometry with flat detector
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        # --- Create Filtered Back-projection (FBP) operator --- #

        # Ray transform (= forward projection).
        self.ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

        # Fourier transform in detector direction
        fourier = odl.trafos.FourierTransform(self.ray_trafo.range, axes=[1])

        # Create ramp in the detector direction
        ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))

        # Create ramp filter via the convolution formula with fourier transforms
        ramp_filter = fourier.inverse * ramp_function * fourier

        # Create filtered back-projection by composing the back-projection (adjoint)
        # with the ramp filter.
        fbp = self.ray_trafo.adjoint * ramp_filter

        fwfcthdl = lambda u: self.ray_trafo(reco_space.element(np.reshape(u, self.domain_dim))).data

        bwfcthdl = lambda f: fbp(np.reshape(f, self.image_dim)).data

        #fwfcthdl = lambda u: 1/self.norm*radon(u, theta=self.theta, circle=False)

        #bwfcthdl = lambda f: iradon(f, theta=self.theta, circle=False, interpolation='linear') #, filter=None
                    # #(2 * len(self.theta)) / np.pi *

        image_dim = np.shape(fwfcthdl(np.ones(domain_dim)))#(int(np.ceil(np.sqrt(2) * max(domain_dim))), len(self.theta))

        #image_dim = (ny, len(self.theta))

        #fwfcthdl = lambda u: np.reshape(R * u.T.ravel(), (len(self.theta), ny)).T

        #bwfcthdl = lambda f: np.reshape(RLop * np.reshape(f, self.image_dim).T.ravel(), self.domain_dim).T

        super(CtRt, self).__init__(domain_dim, image_dim, fwfcthdl, bwfcthdl)

    def matvec(self, x):
        return self*x

    def rmatvec(self, x):
        return self.H*x

    @property
    def H(self):
        a = self.inv
        #a.bwfcthdl = lambda f:  iradon((2 * len(self.theta)) / np.pi * self.norm*f,
        #                                                              theta=self.theta,
        #                                                              circle=False,
        #                                                              filter_name=None,
        #                                                              preserve_range=False,
        #                                                              interpolation="linear")
        a.bwfcthdl = lambda f:  self.ray_trafo.adjoint(np.reshape(f, self.image_dim)).data
        return a
