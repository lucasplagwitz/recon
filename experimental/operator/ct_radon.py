import numpy as np
from skimage.transform import radon, iradon

import sys
#sys.path.append("/Users/lucasplagwitz/git_projects/recon")

from experimental.operator.fcthdl_operator import FcthdlOperator


class CtRt(FcthdlOperator):
    """
    MRI FourierTransform Operator.

    fwfcthdl: RT
    bwfcthdl: IRT
    """

    def __init__(self, domain_dim, center=np.array([0]), theta=None):

        self.domain_dim = domain_dim  # attention: self doppelt...
        self.center = center

        if not any(self.center):
            if len(self.domain_dim) == 2 and self.domain_dim[1] == 1:
                self.center = int((self.domain_dim[0] + 1) / 2)
            else:
                self.center = int((self.domain_dim + 1) / 2)

        # ToDO: input check

        if len(center) == 1:
            self.onevec = 1
        else:
            self.onevec = np.ones(len(self.domain_dim))

        roll_val = self.center - self.onevec

        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.linspace(0., 180., 100, endpoint=False) # max(self.domain_dim)

        fwfcthdl = lambda u: radon(u, theta=self.theta, circle=False)

        bwfcthdl = lambda f: iradon(f, theta=self.theta, circle=False)

        image_dim = (int(np.ceil(np.sqrt(2) * max(domain_dim))), len(self.theta))

        super(CtRt, self).__init__(domain_dim, image_dim, fwfcthdl, bwfcthdl)

