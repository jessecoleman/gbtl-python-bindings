from . import compile_c as c

accumulator = None
semiring = None


class Container(object): pass


class Complement(Container):

    def __init__(self, container):
        self.mat = ~container.mat
        self.shape = container.shape
        self.dtype = container.dtype


class Transpose(Container):

    def __init__(self, container):
        self.mat = container.mat.T()
        self.shape = reversed(container.shape)
        self.dtype = container.dtype


# masked object can be accumulated into
class Masked(object):

    def __init__(self, container, mask=None, replace_flag=False):

        self.container = container
        self.replace_flag = replace_flag

        if mask is None:
            self.mask = c.utilities().NoMask()
            self.mtype = (0, None)
        elif isinstance(mask, Container):
            self.mask = mask.mat
            if isinstance(mask, Complement):
                self.mtype = (2, mask.dtype)
            elif isintance(mask, Transpose):
                self.mtype = (3, mask.dtype)
            else:
                self.mtype = (1, mask.dtype)
        else:
            raise TypeError("Incorrect type for mask parameter")

    def __iadd__(self, other):
        if isinstance(other, Expression):
            return other.eval(self, accumulator)
        
        else:
            return Identity(other).eval(self, accumulator)


class Expression(object):

    def eval(self, C=None, accum=None):

        # construct new output container first
        if C is None:
            out = Masked(self.out())

        # called by expr.eval(C, accum)
        elif isinstance(C, Container):
            out = Masked(C)

        # called by any of C[:], C[0:N], C[M], C[~M]
        elif isinstance(C, Masked):
            out = C

        else:
            raise TypeError("Incorrect type for mask parameter")

        # cpp function call
        return self._call_cpp(out, accum)
 
