from recon.operator.fcthdl_operator import FcthdlOperator


class Identity(FcthdlOperator):

    def __init__(self, domain_dim):

        fwfcthdl = lambda u: u
        bwfcthdl = lambda f: f

        super(Identity, self).__init__(domain_dim=domain_dim,
                                       image_dim=domain_dim,
                                       fwfcthdl=fwfcthdl,
                                       bwfcthdl=bwfcthdl)

    def __mul__(self, other):
        return other

    @property
    def inv(self):
        return self
