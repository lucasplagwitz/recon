import numpy as np

from recon.math.operator.linear_opeartor import LinearOperator

class FcthdlOperator(LinearOperator):

    def __init__(self, domain_dim, image_dim, fwfcthdl, bwfcthdl):

        self.type = 'fcthdlop'
        self.flag = 'regular'
        self.prop = False

        self.domain_dim = domain_dim
        self.image_dim = image_dim
        self.fwfcthdl = fwfcthdl
        self.bwfcthdl = bwfcthdl

    def backwardmult(self, f):
        return self.bwfcthdl(f)

    def forwardmult(self, u):
        return self.fwfcthdl(u)