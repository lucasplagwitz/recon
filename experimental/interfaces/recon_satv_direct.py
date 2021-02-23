from typing import Union

from pylops import Identity, Smoothing2D, Diagonal
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces.satv import SATV

from recon.interfaces import BaseInterface, Recon
from recon.terms import DatanormL2
from recon.solver import PdHgm, PdHgmTGV


class ReconSATVDirect(BaseInterface):
    """
    A Reconstruction object to solve regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        lambda/2 * ||Ax - f||^2 + \alpha(x) J(x)

        J(x) regularisation term J in [TV(), || ||]
    """

    def __init__(self,
                 operator,
                 domain_shape: np.ndarray,
                 assessment: float = 1,
                 noise_sigma: float = 0.2,
                 reg_mode: str = '',
                 lam: float = 0.1,
                 alpha: Union[float, np.ndarray]  = 1,
                 tau: Union[float, str] = None,
                 plot_iteration: bool = False,
                 data_output_path: str = ''):
        self._reg_mode = None

        super(ReconSATVDirect, self).__init__(domain_shape=domain_shape,
                                   reg_mode=reg_mode,
                                   possible_reg_modes=['tv', 'tgv'],
                                   lam=lam,
                                   tau=tau)

        self.operator = operator
        self.solver = None
        self.plot_iteration = plot_iteration
        self.assessment = assessment
        self.noise_sigma = noise_sigma
        self.data_output_path = data_output_path
        self.window_size = 10
        self.w = None
        self.alpha = alpha

        if not isinstance(lam, (int, float)):
            if self.alpha.shape == domain_shape or self.alpha.shape[0] == np.prod(domain_shape):
                #self.alpha = Diagonal(self.alpha.ravel())
                pass
            else:
                msg = "shape of local parameter alpha does not match: "+ \
                      str(self.alpha.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)

    def solve(self, data: np.ndarray, max_iter: int = 400, tol: float = 1e-4):

        super(ReconSATVDirect, self).solve(data=data, max_iter=max_iter, tol=tol)


        plt.Figure()
        u = np.zeros(self.operator.domain_dim)
        k = old_e = 0
        alpha_bar = self.alpha

        pk = 0

        min_alpha = np.min(self.alpha)

        while True:
            print(str(k) + "-Iteration of SATV")
            print(np.linalg.norm(self.operator*u.ravel() - data.ravel(), 2))
            print(self.assessment)
            #print(np.linalg.norm(self.operator*u.ravel() - data.ravel(), 2))
            #print(self.assessment)

            if k > 0:
                # if self.original is None:
                v = (data.ravel() - self.operator*u.ravel())

                # residual filter
                Sop = Smoothing2D(nsmooth=[self.window_size, self.window_size],
                                  dims=self.operator.image_dim, dtype='float64')

                S = np.reshape(Sop * (v ** 2), self.operator.image_dim)
                #T = (self.window_size / self.noise_sigma) ** 2 * S
                #B = (self.noise_sigma / self.window_size) ** 2 * self.noise_sigma
                #B = (self.noise_sigma / self.window_size) ** 2 * (self.noise_sigma ** 2)
                #B = self.noise_sigma**2
                B = self.noise_sigma ** 2 * 1.3
                #S[S < B] = self.noise_sigma ** 2

                #S[S < B] = (self.noise_sigma / self.window_size) ** 2
                S[(S < B)] = self.noise_sigma

                eta = 2  # original == 2
                L = 20
                rho = np.max(alpha_bar) / self.noise_sigma
                #alpha_bar = eta * np.clip(alpha_bar + rho * (np.sqrt(S).ravel() - self.noise_sigma), 0, L)
                alpha_bar = eta * np.clip(alpha_bar + rho * (np.sqrt(S).ravel() - self.noise_sigma), min_alpha, L)
                self.alpha = np.reshape(Sop*alpha_bar.ravel(), self.domain_shape)

                print("MIN: " + str(np.min(self.alpha)))
                print("MAX: "+str(np.max(self.alpha)))

            #self.lam = 1

            rec = Recon(operator=self.operator,
                        domain_shape=self.operator.domain_dim,
                        reg_mode='tv', alpha=(self.lam, 0), lam=self.alpha) #lam=1
            rec.breg_p = pk
            u_new = rec.solve(data=data.ravel(), max_iter=max_iter, tol=1e-4)

            pk = pk - (1 / self.lam) * self.operator.H * (Diagonal(self.alpha.ravel()/2).H * (self.alpha/2) *
                                                          ((self.operator*u_new.ravel() - data.ravel())))

            e = np.linalg.norm(self.operator*u_new.ravel() - data.ravel(), 2)
            if e < self.assessment:
                # which iteration to choose -> norm nearest
                if (old_e - self.assessment) > np.abs(e - self.assessment):
                    u = u_new
                break
            old_e = e
            u = u_new

            k = k + 1

            if self.plot_iteration:
                plt.imshow(np.reshape(u, self.operator.domain_dim), vmin=0, vmax=1)
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(np.reshape(self.alpha, self.operator.image_dim), vmin=0, vmax=1)
                #plt.imshow(np.reshape(self.alpha, self.operator.image_dim))
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '_alpha.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(np.reshape(self.operator.inv*self.alpha.ravel(), self.operator.domain_dim))
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '_recweight.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

        return u
