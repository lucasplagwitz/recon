from typing import Union

from pylops import Identity
import numpy as np
import matplotlib.pyplot as plt

from recon.interfaces.satv import SATV

from recon.interfaces import BaseInterface, Recon
from recon.terms import DatanormL2
from recon.solver.pd_hgm import PdHgm


class ReconSATV(BaseInterface):
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
                 lam: Union[float, np.ndarray] = 0.01,
                 alpha: float = 1.0,
                 tau: Union[float, str] = None,
                 plot_iteration: bool = False,
                 data_output_path: str = ''):
        self._reg_mode = None

        super(ReconSATV, self).__init__(domain_shape=domain_shape,
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
        self.w = None
        self.alpha = alpha

        if not isinstance(lam, (int, float)):
            if self.lam.shape == domain_shape or self.lam.shape[0] == np.prod(domain_shape):
                #self.alpha = Diagonal(self.alpha.ravel())
                pass
            else:
                msg = "shape of local parameter alpha does not match: "+ \
                      str(self.lam.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)

        self.theta = list(np.linspace(0., 60., 60, endpoint=False)) + \
                list(np.linspace(60., 120., 60, endpoint=False)) + \
                list(np.linspace(120., 180., 60, endpoint=True))



    def solve(self, data: np.ndarray, max_iter: int = 1000, tol: float = 1e-4):

        super(ReconSATV, self).solve(data=data, max_iter=max_iter, tol=tol)

        plt.Figure()
        ulast = np.zeros(self.domain_shape)
        u = ulast
        u_sol = u
        k = 0

        w = np.random.normal(size=self.domain_shape)
        w = self.w
        #self.alpha = np.ones(self.domain_shape) * 0.6
        alpha_old = self.alpha


        er = self.operator.inv * np.random.normal(0, self.noise_sigma, size=data.shape).ravel()
        init_sig = noise_sigma2 = np.std(er, ddof=1)
        assessment2 = noise_sigma2*np.sqrt(np.prod(self.domain_shape))

        start_lam = self.lam

        while k< 5: # or (np.linalg.norm(u.ravel() - ulast.ravel())/np.linalg.norm(ulast)) > 0.0001:
            print(np.linalg.norm(u.ravel() - ulast.ravel())/np.linalg.norm(ulast))
            #self.alpha = np.ones(self.domain_shape) * 1
            #self.alpha = (np.ones(self.domain_shape) * 1 + alpha_old) /2
            if k<-1:
                #er = (1-self.lam/np.max(self.lam))*er
                #noise_sigma2 = np.std(er, ddof=1)
                assessment2 = assessment2 - np.linalg.norm(w.ravel() - self.w.ravel(), 2) #noise_sigma2 * np.sqrt(np.prod(self.domain_shape))
                print("......")
                print(noise_sigma2)
                print(assessment2)
                print("......")

            tv_smoothing = SATV(domain_shape=self.domain_shape,
                                reg_mode='tv',
                                lam=self.lam,
                                original=self.w,
                                data_output_path=self.data_output_path,
                                window_size=5,
                                noise_sigma=noise_sigma2,
                                tau='calc',
                                assessment=assessment2)
            u = tv_smoothing.solve(data=w, max_iter=1000, tol=1e-4)
            self.lam = tv_smoothing.lam
            print("--------SATV-FINISHED-------")

            self.lam = self.lam / np.max(self.lam) * 0.01

            rec = Recon(operator=self.operator,
                        domain_shape=self.operator.domain_dim,
                        data=u.ravel(),
                        reg_mode='tik', alpha=(self.lam,0), lam=0.02)
            w = rec.solve(data=data.ravel(), max_iter=400, tol=1e-4)

            if k == 0:
                u0 = u

            k = k + 1

            #self.lam = self.lam / 2

            if self.plot_iteration:
                plt.imshow(np.reshape(u, self.domain_shape), vmin=0, vmax=1)
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(np.reshape(self.lam, self.domain_shape))
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '_alpha.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(np.reshape(w, self.domain_shape), vmin=0, vmax=1)
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '_w.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

        return u



