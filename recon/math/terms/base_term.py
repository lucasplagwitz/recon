import pylops

class BaseTerm(object):

    def __init__(self, O):
        self.tau = 0.99
        self.O = O
        self.data = None

        if isinstance(O, pylops.LinearOperator):
            self.pylops = True

    def set_proxparam(self, tau):
        self.tau = tau

    def get_proxparam(self):
        return self.tau

    def prox(self, f):
        """
        proximal operator of term
        """
        pass