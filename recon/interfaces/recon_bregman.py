import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from pylops import LinearOperator

from recon.terms import DatanormL2Bregman
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
                 operator,
                 domain_shape: np.ndarray,
                 reg_mode: str = 'tv',
                 alpha: float= 1.1,
                 tau: Union[float, str] = None,
                 lam: float = 1,
                 assessment: float = 1,
                 plot_iteration: bool = False,
                 data_output_path: str = ''):

        super(ReconBregman, self).__init__(domain_shape=domain_shape,
                                           reg_mode=reg_mode,
                                           possible_reg_modes=['tv', 'tikhonov', None],
                                           alpha=alpha,
                                           lam=lam,
                                           tau=tau)

        self.operator = operator
        self.plot_iteration = plot_iteration
        self.data_output_path = data_output_path
        self.assessment = assessment

        self.G = DatanormL2Bregman(image_size=domain_shape,
                                   operator=operator,
                                   bregman_weight_alpha=self.alpha,
                                   prox_param=self.tau, lam=self.lam)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        super(ReconBregman, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()

        pk = np.zeros(self.domain_shape)
        pk = pk.ravel()

        if self.plot_iteration:
            plt.Figure()

        ulast = np.zeros(self.domain_shape)
        u = ulast

        i = old_e = 0
        while True:
            print("current norm error: " + str(np.linalg.norm(self.operator * u.ravel() - data.ravel())))
            print("runs till norm <: " + str(self.assessment))

            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.G.pk = pk
            self.G.data = data.ravel()

            self.solver.solve()

            u_new = np.reshape(np.real(self.solver.x), self.domain_shape)

            e = np.linalg.norm(self.operator * u_new.ravel() - data.ravel(), 2)

            if e < self.assessment:
                # which iteration to choose -> norm nearest
                if (old_e - self.assessment) > np.abs(e - self.assessment):
                    u = u_new
                break
            old_e = e

            u = u_new

            pk = pk - (self.lam / self.alpha) * np.real(self.operator.inv*(self.operator*u.ravel() - data.ravel()))
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



