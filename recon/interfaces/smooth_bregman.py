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
        1/2 * ||O*x - f||^2 + \alpha J(x)

        J(x) regularisation term
    """

    def __init__(self,
                 domain_shape: Union[np.ndarray, tuple],
                 reg_mode: str = 'tv',
                 lam: float = 1,
                 alpha: float = 1.1,
                 tau: Union[float, str] = None,
                 assessment: float = 1,
                 plot_iteration: bool = False,
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

        self.G = DatanormL2Bregman(image_size=domain_shape, prox_param=tau, lam=lam)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        super(SmoothBregman, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()

        pk = np.zeros(self.domain_shape)
        pk = pk.ravel()

        if self.plot_iteration:
            plt.Figure()

        ulast = np.zeros(self.domain_shape)
        u = ulast

        i = 0
        while np.linalg.norm(u.ravel() - data.ravel(), 2) > self.assessment:
            print("current norm error: " + str(np.linalg.norm(u.ravel() - data.ravel(), 2)))
            print("runs till norm <: " + str(self.assessment))

            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.G.pk = pk

            self.solver.solve()

            u = np.reshape(np.real(self.solver.var['x']), self.domain_shape)
            pk = pk - (1 / self.alpha) * np.real(u.ravel() - data.ravel())
            i = i + 1

            if self.plot_iteration:
                plt.gray()
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'Bregman_iter' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

        return u


