import numpy as np
from math import log10, sqrt

import odl
from odl.operator.oputils import power_method_opnorm


# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return -1
    max_pixel = np.max(original)
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return round(psnr, 2)

# odl.operator.oputils version of power_method_opnorm for pylops operators

def pylops_power_method_opnorm(pylops_operator, len=256):

    class OdlOperator(odl.Operator):
        def __init__(self, space):

            super(OdlOperator, self).__init__(domain=space, range=space)

        def _call(self, x):
            out = pylops_operator.H * pylops_operator * x
            return out

        def adjoint(self):
            return pylops_operator * pylops_operator.H

    #a = odl.IntervalProd([0]*len, [limit]*len)
    #a = odl.uniform_discr([0]*len, [limit]*len, [100]*len)
    #space = [a]*len
    a = odl.rn(len)
    op = OdlOperator(a)

    # odl.IntervalProd([0], [limit]).element()

    return power_method_opnorm(op)

import pylops
print(pylops_power_method_opnorm(pylops_operator=pylops.Gradient((512, 512),
                                                                 edge=True,
                                                                 dtype='float64', kind='backward'), len=512*512))