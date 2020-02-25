import numpy as np
import copy

from recon.helpers.functions import is_numeric, is_scalar, is_vector


class LinearOperator(object):

    def __init__(self, arg1, arg2=None, flag=""):

        self.prop = False

        self.domain_dim = np.array([1, 1])
        self.image_dim = np.array([1, 1])

        if arg2 is None and flag == "":
            raise NotImplementedError()
        elif all(i is not None for i in [arg1, arg2, flag]):
            self.flag = flag
            if self.flag == 'scalmult':
                pass
            elif self.flag == 'matmult':
                if np.shape(arg1)[1] == np.shape(arg2)[0]:
                    self.arg1 = arg1
                    self.arg2 = arg2
                    self.domain_dim = np.array([np.shape(arg2)[1],0])
                    self.image_dim = np.array([np.shape(arg1)[0], 0])
                else:
                    raise ValueError("Dimensions missmatched!")
                self.flag = 'mult'


        self._shape= 0

    @property
    def shape(self):
        dim = [np.prod(self.image_dim), np.prod(self.domain_dim)]
        return dim

    @property
    def shape_nd(self):
        dim = [self.domain_dim, self.image_dim]
        return dim

    @property
    def T(self):
        obj = copy.copy(self)
        obj.prop = True
        obj.domain_dim = obj.image_dim
        obj.image_dim = self.domain_dim
        if not obj.flag == 'regular':
            obj.arg1 = obj.arg1.T
            obj.arg2 = obj.arg2.T
        return obj

        #self.prop = not self.prop

    def __mul__(self, other):
        if is_scalar(other):
            if self.check_inputs(self, other) == 'linear':
                return LinearOperator(self, other, 'scalmult')
            else:
                # return NonLinearOperator(other, self, 'scalmult')
                pass
        elif is_numeric(other) and is_vector(other):
            if self.flag == 'regular':
                if not self.prop:
                    return np.reshape(self.forwardmult(np.reshape(other, self.shape_nd[0])), self.shape[0])
                else:
                    return np.reshape(self.backwardmult(np.reshape(other, self.shape_nd[0])), self.shape[0])
            elif self.flag == 'mult':
                if not self.prop:
                    return self.arg1 * (self.arg2 * (other))
                else:
                    return self.arg2 * (self.arg1 * (other))
            elif self.flag == 'matadd':
                raise NotImplementedError()
            elif self.flag == 'scaladd':
                raise NotImplementedError()
            elif self.flag == 'mathorzcat':
                raise NotImplementedError()
            elif self.flag == 'matvertcat':
                raise NotImplementedError()

        elif self.shape[1] == other.shape[0]:
            return LinearOperator(self, other, 'matmult')

    def __rmul__(self, other):
        if is_numeric(other) and is_scalar(other):
            return self.__mul__(other)
        else:
            if is_scalar(self):
                if self.check_inputs(other, self) == 'linear':
                    return LinearOperator(other, self, 'scalmult')
                else:
                    # return NonLinearOperator(other, self, 'scalmult')
                    pass
            elif is_numeric(self) and is_vector(self) and self.shape[1] == 1:
                if other.flag == 'regular':
                    if not other.prop:
                        #return np.reshape(self.forwardmult(other, np.reshape(self, ))
                        pass
                elif other.flag == 'mult':
                    if not other.prop:
                        return np.dot(other.arg1, (np.dpt(other.arg2, self)))
                    else:
                        return np.dot(other.arg1, (np.dpt(other.arg2, self)))
                elif other.flag == 'matadd':
                    raise NotImplementedError()
                elif other.flag == 'scaladd':
                    raise NotImplementedError()
                elif other.flag == 'mathorzcat':
                    raise NotImplementedError()
                elif other.flag == 'matvertcat':
                    raise NotImplementedError()

            elif other.shape[1] == self.shape[0]:
                return LinearOperator(other, self, 'matmult')
        return

    @staticmethod
    def check_inputs(op1, op2):
        if issubclass(type(op1), LinearOperator) and issubclass(type(op2), LinearOperator):
            return 'linear'
        else:
            return 'non linear'

    def forwardmult(self, u):
        """
        overwrite
        :param u:
        :return:
        """
        pass