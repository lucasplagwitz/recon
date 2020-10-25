from typing import Union
from pylops import Smoothing2D
import numpy as np
import matplotlib.pyplot as plt

from recon.terms import IndicatorL2, DatanormL2
from recon.interfaces import BaseInterface
from recon.solver.pd_hgm import PdHgm
from recon.solver.pd_hgm_extend import PdHgmTGV


class SATV(BaseInterface):
    """
    A Reconstruction/Smoothing object to solve local adapted.
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
                 alpha: Union[float, tuple] = 1,
                 tau: Union[float, str] = 'calc',
                 window_size: int = 5,
                 data_output_path: str = '',
                 plot_iteration: bool = False):
        self._reg_mode = None

        super(SATV, self).__init__(domain_shape=domain_shape,
                                   reg_mode=reg_mode,
                                   possible_reg_modes=['tv', 'tgv'],
                                   alpha=alpha,
                                   lam=lam,
                                   tau=tau)

        self.domain_shape = domain_shape
        self.reg_mode = reg_mode
        self.solver = None
        self.plot_iteration = plot_iteration
        self.assessment = assessment
        self.noise_sigma = noise_sigma
        self.data_output_path = data_output_path
        self.norm = 1
        self.window_size = window_size

        self.old_lam = 1

        if isinstance(lam, float):
            self.lam = lam * np.ones(domain_shape)
        else:
            self.lam = lam

        self.G = DatanormL2(image_size=domain_shape, prox_param=self.tau, lam=self.lam)

    def solve(self, data: np.ndarray, max_iter: int = 5000, tol: float = 1e-4):

        super(SATV, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()
        self.G.lam = self.lam

        self.F_star.prox_param = self.tau

        plt.Figure()
        u = np.zeros(self.domain_shape)
        k = old_e = 0
        v = data.ravel()
        lam_bar = self.lam
        while True:
            self.old_lam = self.lam
            print(str(k) + "-Iteration of SATV")
            print(np.linalg.norm(u.ravel() - data.ravel()))
            print(self.assessment)

            if k > 0:
                v = (data.ravel() - u.ravel())


                # residual filter
                Sop = Smoothing2D(nsmooth=[self.window_size, self.window_size], dims=self.domain_shape, dtype='float64')

                S = np.reshape(Sop * (v ** 2), self.domain_shape)
                T = (self.window_size / self.noise_sigma) ** 2 * S
                B = (self.noise_sigma / self.window_size) ** 2 * (3 * self.window_size ** 2)
                S[S < B] = self.noise_sigma ** 2  # original implementation with these line

                eta = 2  # original == 2
                L = 100000
                rho = np.max(lam_bar)/self.noise_sigma
                lam_bar = eta * np.clip(lam_bar + rho * (np.sqrt(S).ravel() - self.noise_sigma), 0, L)
                self.lam = np.reshape(Sop*lam_bar.ravel(), self.domain_shape)

                self.G.lam = self.lam.ravel()
                self.G.data = v

            if self.reg_mode == 'tgv':
                self.solver = PdHgmTGV(alpha=self.alpha, lam=self.lam, mode='tv')
                self.solver.max_iter = max_iter
                u_sol = np.reshape(self.solver.solve(np.reshape(v, self.domain_shape)), self.domain_shape)
            else:
                self.solver = PdHgm(self.K, self.F_star, self.G)


                self.solver.max_iter = max_iter
                self.solver.tol = tol

                self.solver.solve()
                u_sol = np.reshape(np.real(self.solver.x), self.domain_shape)

            u_new = u_sol + u

            e = np.linalg.norm(u_new.ravel() - data.ravel(), 2)
            if e < self.assessment:
                # which iteration to choose -> norm nearest
                if (old_e - self.assessment) > np.abs(e - self.assessment):
                    u = u_new
                break
            old_e = e
            u = u_new

            k = k + 1

            if self.plot_iteration:
                plt.gray()
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'SATV_iter' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

        return u
