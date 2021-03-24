from typing import Union
from pylops import Smoothing2D, Smoothing1D
import numpy as np
import matplotlib.pyplot as plt
from recon.operator import Identity

from recon.utils import psnr
from recon.terms import IndicatorL2, DatanormL2
from recon.interfaces import BaseInterface, SmoothBregman
from recon.solver.pd_hgm import PdHgm
from recon.solver.pd_hgm_tgv import PdHgmTGV


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
                 stepsize: float = 2,
                 window_size: int = 10,
                 data_output_path: str = '',
                 plot_iteration: bool = False):
        self._reg_mode = None

        super(SATV, self).__init__(domain_shape=domain_shape,
                                   reg_mode=reg_mode,
                                   possible_reg_modes=['tv', 'tgv'],
                                   alpha=alpha,
                                   lam=lam,
                                   tau=tau)

        self.true_value = None
        self.domain_shape = domain_shape
        self.reg_mode = reg_mode
        self.solver = None
        self.plot_iteration = plot_iteration
        self.assessment = assessment
        self.noise_sigma = noise_sigma
        self.data_output_path = data_output_path
        self.norm = 1
        self.window_size = window_size
        self.bregman = False
        self.stepsize = stepsize
        self.G_template = None

        self.operator = Identity(domain_dim=domain_shape)

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
        min_lam = np.min(self.lam)
        plt.Figure()
        u = np.zeros(self.domain_shape)
        k = old_e = 0
        v = data.ravel()
        lam_bar = self.lam
        while True:
            self.old_lam = self.lam
            print(str(k) + "-Iteration of SATV")
            print(np.linalg.norm((self.operator * u).ravel() - data.ravel()))
            print(self.assessment)

            if k > 0:
                v = (data.ravel() - (self.operator*u).ravel())

                # residual filter
                Sop = Smoothing2D(nsmooth=[self.window_size, self.window_size], dims=self.domain_shape, dtype='float64')

                S = np.reshape(Sop * (v ** 2), self.domain_shape)
                B = self.noise_sigma ** 2 * 1.781576
                S[S < B] = self.noise_sigma ** 2

                if self.bregman:
                    eta = 1
                    rho = 1 / self.noise_sigma
                    l = min_lam
                    L = 10
                else:
                    eta = self.stepsize
                    rho = np.max(lam_bar) / self.noise_sigma
                    l = min_lam
                    L = 40000
                lam_bar = eta * np.clip(lam_bar + rho * (np.sqrt(S).ravel() - self.noise_sigma), l, L)
                self.lam = np.reshape(np.clip(Sop*lam_bar.ravel(), l, L), self.domain_shape)

                if self.G_template is not None:
                    self.G = self.G_template["object"](image_size=self.G_template['image_size'],
                                                       kernel=self.G_template["kernel"],
                                                       cop=self.G_template["kernel"],
                                                       data=np.reshape(data, u.shape), # v
                                                       lam=np.reshape(self.lam, u.shape))
                    self.G.prox_param = self.tau
                else:
                    self.G.lam = self.lam.ravel()
                    self.G.data = data.ravel()

            if self.reg_mode == 'tgv':
                self.solver = PdHgmTGV(alpha=self.alpha, lam=self.lam, mode='tv')
                self.solver.max_iter = max_iter
                u_sol = np.reshape(self.solver.solve(np.reshape(v, self.domain_shape)), self.domain_shape)
            elif not self.bregman:
                self.solver = PdHgm(self.K, self.F_star, self.G)


                self.solver.max_iter = max_iter
                self.solver.tol = tol

                self.solver.solve()
                u_sol = np.reshape(self.solver.x, self.domain_shape)

            if self.bregman:
                ass = self.assessment #np.linalg.norm(u_sol.ravel() - data.ravel() ,2)
                self.lam = self.lam #/ np.max(self.lam)
                alpha = self.alpha #np.mean(self.lam)
                stop_in_front = False #not (k == 4)

                breg_smoothing = SmoothBregman(domain_shape=self.domain_shape,
                                               reg_mode='tv',
                                               alpha=alpha,
                                               lam=self.lam.ravel(),
                                               tau='calc',
                                               plot_iteration=False,
                                               stop_in_front=stop_in_front,
                                               assessment=ass)

                u_sol = breg_smoothing.solve(data=data.ravel(), max_iter=2000, tol=1e-4)


            if self.bregman:
                u_new = u_sol #+ u
            else:
                u_new = u_sol #+ u

            e = np.linalg.norm((self.operator*u_new).ravel() - data.ravel())
            if e < self.assessment:
                # which iteration to choose -> norm nearest
                if np.abs(old_e - self.assessment) > np.abs(e - self.assessment):
                    u = u_new
                if not self.bregman:
                    break
            old_e = e

            if k > 3 and self.bregman:
                break

            u = u_new

            k = k + 1

            if self.plot_iteration:
                #plt.gray()
                plt.close()
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'SATV_iter' + str(k) + '.png', bbox_inches='tight')
                plt.close()

                #plt.gray()
                plt.imshow(np.reshape(self.lam, self.domain_shape), vmin=0, vmax=10)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'SATV_iter_lam' + str(k) + '.png', bbox_inches='tight')
                plt.close()

                plt.close()
                plt.imshow(u, vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(self.data_output_path + 'SATV_iter' + str(k) + '.png', bbox_inches='tight')
                plt.close()

        return u
