import numpy as np

from recon.math.operator.fcthdl_operator import FcthdlOperator

class MriDft(FcthdlOperator):

    def __init__(self, domain_dim, center = 0):


        self.domain_dim = domain_dim # attention: self doppelt...
        self.center = center

        if not any(self.center):
            if len(self.domain_dim) == 2 and self.domain_dim[1] == 1:
                self.center = int((self.domain_dim[0]+1)/2)
            else:
                self.cemter = int((self.domain_dim+1)/2)

        # ToDO: input check

        if len(center) == 1:
            onevec = 1
        else:
            onevec = np.ones(self.domain_dim)

        fwfcthdl = lambda u: np.sqrt(np.prod(self.domain_dim)) * \
                             np.roll(
                                 np.fft.ifftn(
                                     np.fft.ifftshift(u)
                                 ),
                                 -(center - onevec)
                             )

        bwfcthdl = lambda f: 1/np.sqrt(np.prod(self.domain_dim)) * \
                             np.fft.fftshift(
                                 np.fft.fftn(
                                     np.roll(f,
                                             -(-center + onevec)
                                             )
                                 )
                             )
        
        super(MriDft, self).__init__(domain_dim, domain_dim, fwfcthdl, bwfcthdl)

