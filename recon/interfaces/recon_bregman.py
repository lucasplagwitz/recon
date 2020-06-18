import numpy as np
import matplotlib.pyplot as plt
from pylops import LinearOperator

from recon.terms import DatatermBregman
from recon.solver import PdHgm
from recon.interfaces import BaseInterface


class ReconBregman(BaseInterface):
    """
    A Reconstruction object to solve iterative regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        1/2 * ||O*x - f||^2 + \alpha J(x)

        J(x) regularisation term
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

        super(ReconBregman, self).__init__(domain_shape=domain_shape,
                                           reg_mode=reg_mode,
                                           possible_reg_modes=['tv', 'tikhonov', None],
                                           alpha=alpha,
                                           tau=tau)

        self.O = O
        self.plot_iteration = plot_iteration
        self.data_output_path = data_output_path
        self.assessment = assessment

        self.G = DatatermBregman(self.O)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        super(ReconBregman, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.set_proxparam(self.tau)
        self.G.set_proxdata(data.ravel())

        pk = np.zeros(self.domain_shape)
        pk = pk.ravel()

        if self.plot_iteration:
            plt.Figure()

        ulast = np.zeros(self.domain_shape)
        u = ulast

        i = 0
        while np.linalg.norm(self.O * u.ravel() - data.ravel()) > self.assessment:
            print("current norm error: " + str(np.linalg.norm(self.O * u.ravel() - data.ravel())))
            print("runs till norm <: " + str(self.assessment))

            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.G.setP(pk)

            self.solver.solve()

            u = np.reshape(np.real(self.solver.var['x']), self.domain_shape)
            pk = pk - (1 / self.alpha) * np.real(self.O.H*(self.O*u.ravel() - data.ravel()))
            i = i + 1

            if self.plot_iteration:
                plt.gray()
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                #plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'Bregman_iter' + str(i) + '.png', bbox_inches='tight',
                            pad_inches=0)
                plt.close()

        return u



