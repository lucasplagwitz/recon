from pylops.basicoperators import Gradient
from pylops import LinearOperator
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from recon.math.terms import DatatermRecBregman, Projection, DatatermLinear
from recon.math.pd_hgm import PdHgm


class PdReconBregman(object):
    """
    A Reconstruction object to solve regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        1/2 * ||O*x - f||^2 + \alpha J(x)

        J(x) bregman regularisation term
    """

    def __init__(self,
                 O: LinearOperator,
                 domain_shape: np.ndarray,
                 reg_mode: str = 'tv',
                 alpha: float= 1.1,
                 tau: float = None,
                 assessment: float = 1,
                 plot_iteration: bool = False,
                 data_output_path: str = ''):
        self._reg_mode = None

        self.O = O
        self.domain_shape = domain_shape
        self.alpha = alpha
        self.tau = tau
        self.reg_mode = reg_mode
        self.solver = None
        self.plot_iteration = plot_iteration
        self.data_output_path = data_output_path
        self.assessment = assessment

    @property
    def reg_mode(self):
        return self._reg_mode

    @reg_mode.setter
    def reg_mode(self, value):
        if value in ['tv', None]:
            self._reg_mode = value
        else:
            msg = "Please use reg_mode out of ['tv', '']"
            raise ValueError(msg)

    def solve(self, data: np.ndarray, maxiter: int = 150, tol: float = 5*10**(-4)):

        if self.reg_mode is not None:
            grad = Gradient(dims=self.domain_shape, edge = True, dtype='float64', kind='backward')
            K = self.alpha * grad
            if not self.tau:
                norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                sigma = 0.99 / norm
                print("Calced tau: "+str(sigma) + ". "
                      "Next run with same alpha, set this tau value to decrease runtime.")
                tau = sigma
            else:
                tau = self.tau
                sigma = tau
            if self.reg_mode == 'tv':
                F_star = Projection(self.domain_shape, len(self.domain_shape))
            else:
                F_star = DatatermLinear()
                F_star.set_proxdata(0)
        else:
            tau = 0.99
            sigma = tau
            F_star = DatatermLinear()
            K = 0

        G = DatatermRecBregman(self.O)
        G.set_proxparam(tau)
        G.set_proxdata(data.ravel())
        F_star.set_proxparam(sigma)

        pk = np.zeros(self.domain_shape)
        pk = pk.T.ravel()
        plt.Figure()
        ulast = np.zeros(self.domain_shape)
        u01 = ulast

        i = 0
        while np.linalg.norm(self.O * u01.ravel() - data.ravel()) > self.assessment:
            print("norm error: " + str(np.linalg.norm(self.O * u01.ravel() - data.ravel())))

            self.solver = PdHgm(K, F_star, G)
            self.solver.maxiter = maxiter
            self.solver.tol = tol

            G.set_proxdata(data.ravel())
            G.setP(pk)
            self.solver.solve()
            u01 = np.reshape(np.real(self.solver.var['x']), self.domain_shape)
            pk = pk - (1 / self.alpha) * np.real(self.O.H*(self.O*u01.ravel() - data.ravel()))
            i = i + 1
            if self.plot_iteration:
                plt.gray()
                plt.imshow(u01, vmin=0, vmax=1)
                plt.axis('off')
                #plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'Bregman_reconstruction_iter' + str(i) + '.png', bbox_inches='tight',
                            pad_inches=0)
                plt.close()

        return np.reshape(self.solver.var['x'], self.domain_shape)



