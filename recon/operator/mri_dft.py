import numpy as np

from recon.operator.fcthdl_operator import FcthdlOperator

class MriDft(FcthdlOperator):
    """
    MRI FourierTransform Operator.

    fwfcthdl: sqrt(m*n) * FFT
    bwfcthdl: 1/sqrt(m*n) * IFFT
    """

    def __init__(self, domain_dim, center=np.array([0])):


        self.domain_dim = domain_dim # attention: self doppelt...
        self.center = center

        if not any(self.center):
            if len(self.domain_dim) == 2 and self.domain_dim[1] == 1:
                self.center = int((self.domain_dim[0]+1)/2)
            else:
                self.center = np.array([x//2 for x in list(domain_dim)])

        # ToDO: input check

        if len(center) == 1:
            self.onevec = 1
        else:
            self.onevec = np.ones(len(self.domain_dim))

        roll_val = self.center - self.onevec

        fwfcthdl = lambda u: np.sqrt(np.prod(self.domain_dim)) * \
                             np.roll(
                                 np.roll(
                                     np.fft.ifftn(
                                         np.fft.ifftshift(u)
                                     ),
                                     -int(roll_val[0]), axis= 0
                                 ), -int(roll_val[1]), axis= 1
                             )

        bwfcthdl = lambda f: np.real(1/np.sqrt(np.prod(self.domain_dim)) * \
                                         np.fft.fftshift(
                                             np.fft.fftn(
                                                 np.roll(
                                                     np.roll(f,
                                                             int(roll_val[0]), axis=0
                                                             ), int(roll_val[1]), axis= 1
                                                 )
                                             )
                                         )
                                     )
        
        super(MriDft, self).__init__(domain_dim, domain_dim, fwfcthdl, bwfcthdl)

    @property
    def H(self):
        return self.inv