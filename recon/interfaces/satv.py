from typing import Union
from pylops import Smoothing2D
import numpy as np
import matplotlib.pyplot as plt

from recon.terms import IndicatorL2, DatanormL2
from recon.interfaces import BaseInterface
from recon.solver.pd_hgm import PdHgm


class SATV(BaseInterface):
    """
    A Reconstruction object to solve regularized inverse reconstruction problems.
    Solver is Primal-Dual based.

    Form:
        1/2 * ||x - f||^2 + \alpha TV(x)

    """

    def __init__(self,
                 domain_shape: Union[np.ndarray, tuple],
                 assessment: float = 1,
                 noise_sigma: float = 0.2,
                 reg_mode: str = 'tv',
                 lam: Union[float, np.ndarray] = 0.01,
                 tau: float = None,
                 data_output_path: str = '',
                 plot_iteration: bool = False):
        self._reg_mode = None

        super(SATV, self).__init__(domain_shape=domain_shape,
                                   reg_mode=reg_mode,
                                   possible_reg_modes=['tv'],
                                   alpha=1,
                                   lam=lam,
                                   tau=tau)

        self.domain_shape = domain_shape
        self.tau = tau
        self.reg_mode = reg_mode
        self.solver = None
        self.plot_iteration = plot_iteration
        self.assessment = assessment
        self.noise_sigma = noise_sigma
        self.data_output_path = data_output_path
        self.norm = 1

        if isinstance(lam, float):
            self.lam = lam * np.ones(domain_shape)
        else:
            self.lam = lam

        self.G = DatanormL2(image_size=domain_shape, prox_param=tau)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 6*10**(-4)):

        super(SATV, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()
        self.G.lam = self.lam


        self.F_star.prox_param = self.tau

        plt.Figure()
        ulast = np.zeros(self.domain_shape)
        u = ulast
        k = 0
        lam_bar = self.lam
        while np.linalg.norm(u.ravel() - data.ravel(), 2) > self.assessment:
            print(str(k) + "-Iteration of SATV")
            print(np.linalg.norm(u.ravel() - data.ravel()))
            print(self.assessment)

            if k > 0:
                v = (data.ravel() - u.ravel())
                self.G.lam = self.lam.ravel()
                self.G.data = v

                # residual filter
                w = 10
                Sop = Smoothing2D(nsmooth=[w, w], dims=self.domain_shape, dtype='float64')

                S = np.reshape(Sop * (v ** 2), self.domain_shape)
                T = (w / self.noise_sigma) ** 2 * S
                B = (self.noise_sigma / w) ** 2 * (3 * w ** 2)
                #S[S < B] = self.noise_sigma ** 2  # original implementation with these line

                eta = 1.5  # original == 2
                L = 100000
                rho = np.max(lam_bar)/self.noise_sigma
                lam_bar = eta * np.clip(lam_bar + rho * (np.sqrt(S).ravel() - self.noise_sigma), 0, L)
                self.lam = np.reshape(Sop*lam_bar.ravel(), self.domain_shape)

            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol

            self.solver.solve()
            u_sol = np.reshape(np.real(self.solver.var['x']), self.domain_shape)
            u = u_sol + u
            k = k + 1

            if self.plot_iteration:
                plt.gray()
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'SATV_iter' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

        return u
