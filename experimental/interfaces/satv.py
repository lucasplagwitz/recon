from pylops.basicoperators import Gradient
from pylops import Diagonal, Smoothing2D, BlockDiag, Smoothing1D
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

from experimental.terms.dataterm_linear_constrained import DatatermLinearConstrained

from recon.terms import Projection, DatatermLinear
from recon.solver.pd_hgm import PdHgm


class SATV(object):
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
        self.norm = 1

        if type(alpha) is not float:
            if self.alpha.shape == domain_shape:
                #self.alpha = Diagonal(self.alpha.ravel())
                pass
            else:
                msg = "shape of local parameter alpha does not match: "+ \
                      str(self.alpha.shape) + "!=" + str(domain_shape)
                raise ValueError(msg)


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

            #if not isinstance(self.alpha, float):
            K = grad
            #else:
            #    K = self.alpha * grad

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
        G.set_weight(self.alpha.ravel())
        F_star.set_proxparam(sigma)

        plt.Figure()
        ulast = np.zeros(self.domain_shape)
        u = ulast
        u_sol = u
        k = 0


        while np.linalg.norm(u.ravel() - data.ravel()) > self.assessment:
            print(str(k) + "-Iteration of SATV")
            print(np.linalg.norm(u.ravel() - data.ravel()))
            print(self.assessment)

            # G.set_proxparam(tau)
            F_star.set_proxparam(sigma)

            if k == 0:
                alpha_bar = self.alpha
            else:
                v = (data.ravel() - u.ravel())
                G = DatatermLinear()
                G.set_proxdata(v)
                G.set_proxparam(tau)
                G.set_weight(self.alpha.ravel())

                # residual filter
                w = 20
                if len(self.domain_shape) == 1:
                    Sop = Smoothing1D(nsmooth=w, dims=self.domain_shape, dtype='float64')
                elif len(self.domain_shape) == 2:
                    Sop = Smoothing2D(nsmooth=[w, w], dims=self.domain_shape, dtype='float64')
                else:
                    raise ValueError("domain_shape Smoothing operator.")
                S = np.reshape(Sop * (v ** 2), self.domain_shape)
                T = (w / self.noise_sigma) ** 2 * S
                B = (self.noise_sigma / w) ** 2 * (3 * w ** 2)
                S[S < B] = self.noise_sigma ** 2

                # S = np.clip(S, self.noise_sigma**2,5)
                eta = 2  # 0.7
                L = 100000
                #rho = np.max(alpha_bar)/self.noise_sigma
                rho = 0  # 5 0.9
                alpha_bar = eta * np.clip(alpha_bar + rho * (np.sqrt(S) - self.noise_sigma), 0.0001, L)
                #print(np.max(alpha_bar))
                #Sop = Smoothing2D(nsmooth=[2, 2], dims=self.domain_shape, dtype='float64')
                self.alpha = np.reshape(alpha_bar.ravel(), self.domain_shape)
                # grad = Gradient(dims=self.domain_shape, edge=True, dtype='float64', kind="backward")
                # K = BlockDiag([Diagonal(self.alpha.ravel()), Diagonal(self.alpha.ravel())]) * grad
                # norm = np.abs(np.asscalar(K.eigs(neigs=1, which='LM')))
                # sigma = 0.99 / norm
                # sigma = self.tau
                # tau = sigma

            self.solver = PdHgm(K, F_star, G)
            self.solver.maxiter = maxiter
            self.solver.tol = tol

            G.set_proxparam(tau)
            F_star.set_proxparam(sigma)

            self.solver.solve()
            u_sol = np.reshape(np.real(self.solver.var['x']), self.domain_shape)
            u = u_sol + u
            ulast = u
            k = k + 1

            if self.plot_iteration:
                #plt.gray()
                if len(self.domain_shape)==2:
                    plt.imshow(u)
                    plt.axis('off')
                    # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                    plt.savefig(self.data_output_path + 'SATV_iter' + str(k) + '.png',
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    #plt.gray()
                    plt.imshow(self.alpha)
                    plt.axis('off')
                    # plt.title('RRE =' + str(round(RRE_breg, 2)), y=-0.1, fontsize=20)
                    plt.savefig(self.data_output_path + 'SATV_iter' + str(k) + '_alpha.png',
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                else:
                    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
                    ax1.plot(list(range(len(u))), list(u))
                    ax2.plot(list(range(len(u))), list(self.alpha))
                    ax3.plot(list(range(len(u))), list(u - data))
                    plt.savefig(self.data_output_path + 'SATV_AUDIO_iter' + str(k) + '_alpha.png',
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    solution = (2600/np.max(u)*u).astype(np.int16)
                    audio_segment = AudioSegment(
                        solution.tobytes(),
                        frame_rate=44100,
                        sample_width=solution.dtype.itemsize,
                        channels=1
                    )
                    audio_segment.export("/Users/lucasplagwitz/Desktop/audio/iter"+str(k)+".wav", format="wav")

        return u



