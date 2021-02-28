from typing import Union
import numpy as np
import matplotlib.pyplot as plt

from recon.terms import DatanormL2Bregman
from recon.solver import PdHgm
from recon.interfaces import BaseInterface


class SmoothBregman(BaseInterface):
    """
    A Reconstruction object to solve iterative regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        lam/2 * ||O*x - f||^2 + alpha J(x)

        J(x) regularisation term
    """

    def __init__(self,
                 domain_shape: Union[np.ndarray, tuple],
                 reg_mode: str = 'tv',
                 lam: float = 1,
                 alpha: float = 1,
                 tau: Union[float, str] = 'calc',
                 assessment: float = 1,
                 plot_iteration: bool = False,
                 stop_in_front: bool = False,
                 data_output_path: str = ''):

        super(SmoothBregman, self).__init__(domain_shape=domain_shape,
                                            reg_mode=reg_mode,
                                            possible_reg_modes=['tv', 'tikhonov', None],
                                            lam=lam,
                                            alpha=alpha,
                                            tau=tau)

        self.plot_iteration = plot_iteration
        self.data_output_path = data_output_path
        self.assessment = assessment
        self.stop_in_front = stop_in_front

        self.G = DatanormL2Bregman(image_size=domain_shape,
                                   prox_param=self.tau,
                                   lam=lam,
                                   bregman_weight_alpha=self.alpha
                                   )

    def solve(self, data: np.ndarray, max_iter: int = 5000, tol: float = 1e-4):

        super(SmoothBregman, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()

        pk = np.zeros(self.domain_shape).ravel()

        if self.plot_iteration:
            plt.Figure()

        u = np.zeros(self.domain_shape)

        i = old_e = 0
        while True:
            print("current norm error: " + str(np.linalg.norm(u.ravel() - data.ravel())))
            print("runs till norm <: " + str(self.assessment))

            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.G.pk = pk

            self.solver.solve()

            u_new = np.reshape(self.solver.x, self.domain_shape)

            e = np.linalg.norm(u_new.ravel() - data.ravel())
            if e < self.assessment:
                # which iteration to choose -> norm nearest
                if np.abs(old_e-self.assessment) > np.abs(e-self.assessment) and not self.stop_in_front:
                    u = u_new
                break
            old_e = e

            u = u_new

            #pk = pk - (self.lam / self.alpha) * (u.ravel() - data.ravel())
            pk = pk - self.lam * (u.ravel() - data.ravel())
            i = i + 1

            if self.plot_iteration:
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'Bregman_iter' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

        return u
