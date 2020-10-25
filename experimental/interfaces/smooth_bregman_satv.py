from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from pylops import Smoothing2D

from recon.terms import DatanormL2Bregman
from recon.solver import PdHgm
from recon.interfaces import BaseInterface


class SmoothBregmanSATV(BaseInterface):
    """
    A local smoothing object to solve iterative regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        1/2 * ||x - f||^2 + \alpha D_TV(x)

    """

    def __init__(self,
                 domain_shape: Union[np.ndarray, tuple],
                 reg_mode: str = 'tv',
                 lam: Union[float, np.ndarray] = 1,
                 alpha: float = 1,
                 tau: Union[float, str] = None,
                 assessment: float = 1,
                 plot_iteration: bool = False,
                 noise_sigma: float = 0.1,
                 data_output_path: str = ''):

        super(SmoothBregmanSATV, self).__init__(domain_shape=domain_shape,
                                            reg_mode=reg_mode,
                                            possible_reg_modes=['tv', 'tikhonov', None],
                                            lam=lam,
                                            alpha=alpha,
                                            tau=tau)

        self.plot_iteration = plot_iteration
        self.data_output_path = data_output_path
        self.assessment = assessment
        self.noise_sigma = noise_sigma

        if isinstance(lam, (float, int)):
            self.lam = lam * np.ones(domain_shape)
        else:
            self.lam = lam

        self.G = DatanormL2Bregman(image_size=domain_shape, prox_param=self.tau, lam=self.lam,
                                   bregman_weight_alpha=self.alpha)

    def solve(self, data: np.ndarray, max_iter: int = 150, tol: float = 5*10**(-4)):

        super(SmoothBregmanSATV, self).solve(data=data, max_iter=max_iter, tol=tol)

        self.G.data = data.ravel()

        pk = np.zeros(self.domain_shape)
        pk = pk.ravel()

        if self.plot_iteration:
            plt.Figure()

        ulast = np.zeros(self.domain_shape)
        u = ulast

        k = old_e = 0
        lam_bar = self.lam
        while True:
            print("current norm error: " + str(np.linalg.norm(u.ravel() - data.ravel(), 2)))
            print("runs till norm <: " + str(self.assessment))

            if k > 0:
                v = (data.ravel()- u.ravel())


                # residual filter
                w = 5
                Sop = Smoothing2D(nsmooth=[w, w], dims=self.domain_shape, dtype='float64')

                S = np.reshape(Sop * (v ** 2), self.domain_shape)
                T = (w / self.noise_sigma) ** 2 * S
                B = (self.noise_sigma / w) ** 2 * (3 * w ** 2)
                S[S < B] = self.noise_sigma ** 2  # original implementation with these line

                eta = 2 # original == 2
                L = 100000
                rho = np.max(lam_bar) / self.noise_sigma
                lam_bar = eta * np.clip(lam_bar + rho * (np.sqrt(S).ravel() - self.noise_sigma), 0, L)
                self.lam = np.reshape(Sop*lam_bar.ravel(), self.domain_shape)

                self.G.lam = self.lam.ravel()
                v = data.ravel()
                self.G.data = v
            else:
                v = data.ravel()

            self.solver = PdHgm(self.K, self.F_star, self.G)
            self.solver.max_iter = max_iter
            self.solver.tol = tol
            self.G.pk = pk

            self.solver.solve()

            u_new = np.reshape(self.solver.var['x'], self.domain_shape)

            e = np.linalg.norm(u_new.ravel() - data.ravel(), 2)

            if e < self.assessment:
                # which iteration to choose -> norm nearest
                if (old_e - self.assessment) > np.abs(e - self.assessment):
                    u = u_new
                break
            old_e = e

            u = u_new

            pk = pk - (self.lam / self.alpha) * (u.ravel() - v)
            k += 1

            if self.plot_iteration:
                plt.gray()
                plt.imshow(u)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'Bregman_iter' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.gray()
                plt.imshow(np.reshape(self.lam.ravel(), self.domain_shape))
                plt.axis('off')
                plt.savefig(self.data_output_path + 'lam_Bregman_iter' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

        return u
