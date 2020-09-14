from pylops.basicoperators import Gradient
from pylops import Diagonal
import numpy as np
import matplotlib.pyplot as plt

from experimental.operator.ct_radon import CtRt
from recon.interfaces.satv import SATV

from recon.terms import Projection, DatatermLinear, Dataterm
from recon.solver.pd_hgm import PdHgm


class ReconSATV(object):
    """
    A Reconstruction object to solve regularized inverse reconstruction problems.
    Solver is Primal-Dual based.
    Form:
        1/2 * ||x - f||^2 + \alpha J(x)

        J(x) regularisation term J in [TV(), || ||]
    """

    def __init__(self,
                 domain_shape: np.ndarray,
                 assessment: float = 1,
                 noise_sigma: float = 0.2,
                 reg_mode: str = '',
                 alpha=0.01,
                 tau: float = None,
                 data_output_path: str = ''):
        self._reg_mode = None

        self.domain_shape = domain_shape
        self.alpha = alpha
        self.tau = tau
        self.reg_mode = reg_mode
        self.solver = None
        self.plot_iteration = True
        self.assessment = assessment
        self.noise_sigma = noise_sigma
        self.data_output_path = data_output_path
        self.w = None

        if type(alpha) is not float:
            if self.alpha.shape == domain_shape:
                #self.alpha = Diagonal(self.alpha.ravel())
                pass
            else:
                msg = "shape of local parameter alpha does not match: "+ \
                      str(self.alpha.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)

        self.theta = list(np.linspace(0., 60., 60, endpoint=False)) + \
                list(np.linspace(60., 120., 60, endpoint=False)) + \
                list(np.linspace(120., 180., 60, endpoint=True))


    @property
    def reg_mode(self):
        return self._reg_mode

    @reg_mode.setter
    def reg_mode(self, value):
        if value in ['tikhonov', 'tv', None]:
            self._reg_mode = value
        else:
            msg = "Please use reg_mode out of ['tikhonov', 'tv', '']"
            raise ValueError(msg)

    def solve(self, data: np.ndarray, maxiter: int = 150, tol: float = 5*10**(-4)):

        if self.reg_mode is not None:

            grad = Gradient(dims=self.domain_shape, edge = True, dtype='float64', kind="backward")

            if not isinstance(self.alpha, float):
                # K = BlockDiag([Diagonal(self.alpha.ravel()), Diagonal(self.alpha.ravel())]) * grad
                K = grad
            else:
                K = self.alpha * grad

            if not self.tau:
                if np.prod(self.domain_shape) > 25000:
                    long = True
                else:
                    long = False
                if long:
                    print("Start evaluate tau. Long runtime.")

                norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                #norm = 0.001
                sigma = 0.99 / norm
                if long:
                    print("Calc tau: "+str(sigma))
                tau = sigma
            else:
                tau = self.tau
                sigma = tau

            if self.reg_mode == 'tv':
                F_star = Projection(self.domain_shape, len(self.domain_shape))
            else:
                F_star = DatatermLinear()
                F_star.set_proxdata(0)
        else:
            tau = 0.99
            sigma = tau
            F_star = DatatermLinear()
            K = 0

        G = DatatermLinear()
        G.set_proxparam(tau)
        G.set_proxdata(data.ravel())
        F_star.set_proxparam(sigma)

        plt.Figure()
        ulast = np.zeros(self.domain_shape)
        u = ulast
        u_sol = u
        k = 0

        #w = np.random.normal(size=self.domain_shape)
        w = self.w
        #self.alpha = np.ones(self.domain_shape) * 0.6
        alpha_old = self.alpha

        while k< 5: # or (np.linalg.norm(u.ravel() - ulast.ravel())/np.linalg.norm(ulast)) > 0.0001:
            print(np.linalg.norm(u.ravel() - ulast.ravel())/np.linalg.norm(ulast))
            self.alpha = np.ones(self.domain_shape) * 1
            #self.alpha = (np.ones(self.domain_shape) * 1 + alpha_old) /2
            tv_smoothing = SATV(domain_shape=w.shape, reg_mode='tv', alpha=self.alpha,
                                data_output_path=self.data_output_path, noise_sigma=self.noise_sigma,
                                tau=0.00035,
                                assessment=self.assessment)
            ulast = u
            u = tv_smoothing.solve(data=w, maxiter=150, tol=5 * 10 ** (-6))

            self.alpha = tv_smoothing.alpha
            alpha_old = self.alpha

            operator = CtRt(np.shape(w),
                                 np.array([(np.shape(w)[0] / 2) + 1, (np.shape(w)[0] / 2) + 1]),
                                 theta=self.theta)

            #norm = 0.99 / np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
            #norm = 0.99 / np.max(self.alpha)

            fac = 1.0
            K = fac * Diagonal((self.alpha.ravel()/np.max(self.alpha)))
            #K = fac*Identity(np.prod(self.domain_shape))
            #tau = 0.99 / np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
            G = Dataterm(operator)
            #tau = 0.99 / fac
            #G.faktor = 0.5
            F_star = DatatermLinear()
            F_star.set_proxdata(-1*u.ravel())
            #F_star.set_weight(fac)
            F_star.set_proxparam(tau)

            G.set_proxparam(tau)
            G.set_proxdata(data.ravel())
            solver = PdHgm(K, F_star, G)
            solver.max_iter = 250
            solver.tol = tol
            solver.solve()
            w = np.reshape(solver.var['x'], self.domain_shape)


            """
            if k == 0:
                alpha_bar = self.alpha
                G = DatatermLinear()
                G.set_proxparam(tau)
                G.set_proxdata(data.ravel())
            else:
                v = (data.ravel() - u.ravel())
                G = DatatermLinear()
                G.set_proxdata(v)

                # residual filter
                w = 6
                Sop = Smoothing2D(nsmooth=[w, w], dims=self.domain_shape, dtype='float64')
                S = np.reshape(Sop * (v**2), self.domain_shape)
                T = (w/self.noise_sigma)**2 * S
                B = (self.noise_sigma/w) ** 2 * (3*w**2)
                S[S<B] = self.noise_sigma**2

                #S = np.clip(S, self.noise_sigma**2,5)
                eta = 0.7
                L = 1000
                #rho = np.max(alpha_bar)/self.noise_sigma
                rho = 0.9
                alpha_bar = eta * np.clip(alpha_bar - rho*(np.sqrt(S) - self.noise_sigma), 1/1000, L)
                self.alpha = np.reshape(Sop*alpha_bar.ravel(), self.domain_shape)
                grad = Gradient(dims=self.domain_shape, edge=True, dtype='float64', kind="backward")
                K = BlockDiag([Diagonal(self.alpha.ravel()), Diagonal(self.alpha.ravel())]) * grad
                norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                sigma = 0.99 / norm
                tau = sigma

            self.solver = PdHgm(K, F_star, G)
            self.solver.maxiter = maxiter
            self.solver.tol = tol

            G.set_proxparam(tau)
            F_star.set_proxparam(sigma)

            self.solver.solve()
            u_sol = np.reshape(np.real(self.solver.var['x']), self.domain_shape)
            u = u_sol + u
            ulast = u
            """
            k = k + 1

            if self.plot_iteration:
                plt.imshow(u, vmin=0, vmax=0.5)
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(self.alpha)
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '_alpha.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(w, vmin=0, vmax=0.5)
                plt.axis('off')
                # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                plt.savefig(self.data_output_path + 'RESATV_segmentation2_iter' + str(k) + '_w.png',
                            bbox_inches='tight',
                            pad_inches=0)
                plt.close()

        return u



