import numpy as np
import copy
import scipy.sparse.linalg


class PdHgm(object):

    def __init__(self, K, F_star, G):
        self.K = K
        self.F_star = F_star
        self.G = G
        self.maxiter = 500
        self.tol = 10**(-4)
        self.k = 1
        self.sens = 0.001

        self.plot_on = False

        self.var = {'x': None, 'y': None}

    def restart_counter(self):
        self.k = 1

    def initialise(self):
        self.k = 1
        self.res = np.inf
        self.resold = 1
        self.sens = 0.001
        self.var['x'] = np.zeros(self.K.shape[1])
        self.var['y'] = np.zeros(self.K.shape[0])

    def solve(self):

        if self.var['x'] is None and self.var['y'] is None:
            self.initialise()

        if self.plot_on:
            raise NotImplementedError()

        while (self.tol < self.sens or self.k == 1) and (self.k <= self.maxiter):
            if self.k % 20 == 0:
                print(self.k)
            self.var_prev = copy.copy(self.var)

            self.var['x'] = self.G.prox(self.var['x'] -
                                        self.G.get_proxparam() * (self.K.T*self.var['y'])
                                        )

            self.var['y'] = self.F_star.prox(self.var['y'] + self.F_star.get_proxparam() *
                                                (self.K * (2 * self.var['x'] - self.var_prev['x']))
                                            )


            self.update_sensivity()

            self.k += 1

    def update_sensivity(self):
        aux1 = self.var['x'] - self.var_prev['x']
        aux2 = self.var['y'] - self.var_prev['y']
        self.sens = 1 / 2 * (np.linalg.norm(aux1 - self.G.get_proxparam() * (self.K.T * aux2), 2)/
                                                        np.linalg.norm(self.var['x'], 2) +
                                                    np.linalg.norm(aux2 - self.F_star.get_proxparam() * (self.K * aux1), 2) /
                                                    np.linalg.norm(self.var['y'], 2))
