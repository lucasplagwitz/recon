import numpy as np
import copy

from recon.utils.functions import is_numeric, is_scalar, is_vector, switch_arguments


class LinearOperator(object):
    """
    Abstract Class for object operator: X, Y Banachspaces:
    O : X -> Y linear (and bounded?).
        X from R^{n_1 x n_2}
        Y from R^{m_1 x m_2}
    2D only supported.
    """
    def __init__(self, arg1, arg2=None, flag=""):
        """


        important values:
        domain_dim: (= dim(X)) -> araay (n_1, n_2)
            example: scalar_prod (not sure if Operator): (1, 1)
                     radon_transformation on 256x256 px (256, 256)
        image_dim: (= dim(Y)) -> array (m_1, m_2)

        :param arg1: type: LineareOperator, np.array, Scalar
        :param arg2: type: LineareOperator, np.array, Scalar
        :param flag: operation
        """

        self.prop = False

        self.domain_dim = np.array([1, 1])
        self.image_dim = np.array([1, 1])

        if arg2 is None and flag == "":
            raise NotImplementedError()
        elif all(i is not None for i in [arg1, arg2, flag]):
            self.flag = flag

            # flag as multiplication
            if self.flag == 'scalmult':
                """
                    O[arg1, arg2](p) = (arg1*arg2)(p)  (second argument always scalar)
                        => O.domain_dim = colums(arg1) = rows(p)  
                        => O.image_dim = rows(arg1)
                """
                if is_scalar(arg1):
                    arg1, arg2 = switch_arguments(arg1, arg2)
                elif not is_scalar(arg2):
                    raise ValueError("ScalMult requires one scalar value.")
                self.arg1 = arg1
                self.arg2 = arg2
                self.domain_dim = arg1.shape[1]
                self.image_dim = arg1.shape[0]
                self.flag = 'mult'

            elif self.flag == 'matmult':
                """
                O[arg1, arg2](p) = (arg1*arg2)(p) 
                    => O.domain_dim = colums(arg2) = rows(p)  
                    => O.image_dim = rows(arg1)
                """
                if np.shape(arg1)[1] == np.shape(arg2)[0]:
                    self.arg1 = arg1
                    self.arg2 = arg2
                    self.domain_dim = np.array([np.shape(arg2)[1], 1])
                    self.image_dim = np.array([np.shape(arg1)[0], 1])
                else:
                    raise ValueError("Dimensions missmatched!")
                self.flag = 'mult'

            # flag as addition
            elif self.flag == 'scaladd':
                """
                O[arg1, arg2](p) = (arg1 + arg2)(p) (second argument always scalar)
                    => O.domain_dim = colums(arg1) = rows(p)  
                    => O.image_dim = rows(arg1)
                """
                if is_scalar(arg1):
                    arg1, arg2 = switch_arguments(arg1, arg2)
                elif not is_scalar(arg2):
                    raise ValueError("ScalAdd requires one scalar value.")
                self.domain_dim = arg1.shape[1]
                self.image_dim = arg1.shape[0]
            elif self.flag == 'matadd':
                """
                    O[arg1, arg2](p) = (arg1+arg2)(p) 
                        => O.domain_dim = colums(arg1) = colums(arg2) = rows(p)  
                        => O.image_dim = rows(arg1) = rows(arg2)
                """
                if arg2.shape == arg1.shape:
                    self.domain_dim = arg1.shape[1]
                    self.image_dim = arg1.shape[0]
                else:
                    raise ValueError("Dimension Error.")
            elif self.flag:
                raise NotImplementedError("This operations not implemented yet...")

        self._shape = 0

    @property
    def shape(self):
        dim = [np.prod(self.image_dim), np.prod(self.domain_dim)]
        return dim

    @property
    def shape_nd(self):
        dim = [self.domain_dim, self.image_dim]
        return dim

    @property
    def inv(self):
        obj = copy.copy(self)
        obj.prop = True
        obj.domain_dim = obj.image_dim
        obj.image_dim = self.domain_dim
        if not obj.flag == 'regular':
            if not is_scalar(obj.arg1):
                obj.arg1 = obj.arg1.T
            if not is_scalar(obj.arg2):
                obj.arg2 = obj.arg2.T
        return obj

        #self.prop = not self.prop

    # multiplication
    def __mul__(self, other):
        if is_scalar(other):
            if self.check_inputs(self, other) == 'linear':
                return LinearOperator(self, other, 'scalmult')
            else:
                # return NonLinearOperator(other, self, 'scalmult')
                raise NotImplementedError("no nonlineare Operator")
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

    def __add__(self, other):
        return

    @staticmethod
    def check_inputs(op1, op2):
        # Todo Nonlineare
        if issubclass(type(op1), NonLinearOperator) or issubclass(type(op2), NonLinearOperator):
            return 'non linear'
        else:
            return 'linear'

    def forwardmult(self, u):
        """
        overwrite
        :param u:
        :return:
        """
        pass

    def backwardmult(self, f):
        pass


class NonLinearOperator(object):

    pass
