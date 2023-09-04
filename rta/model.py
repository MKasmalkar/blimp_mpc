import numpy as np

class Model:
    def __init__(self, linear=True, control_affine=True):
        self.linear = linear
        self.control_affine = control_affine
        self.jacobian = lambda: None # df / dx
        self.input_gradient = lambda: None # dg / dx
        self.A = None
        self.B = None
        self.C = None # optional
        self.shape = (0, 0)

    def A_(self, A):
        if isinstance(A, np.ndarray) and self.linear:
            self.A = A
            self.shape_(A.shape)
    
    def B_(self, B):
        if isinstance(B, np.ndarray) and self.control_affine:
            self.B = B

    def C_(self, C):
        if isinstance(C, np.ndarray):
            self.C = C
    
    def shape_(self, shape):
        self.shape = shape
    
    def jacobian_(self, jac):
        self.jacobian_hidden = jac

    #@abstractmethod
    def jacobian(self, x):
        pass
    
    #@abstractmethod
    def compute_LTV(self, x0, u):
        # Get a list of matrix pairs (A,B) given an initial condition and an input sequence.
        pass
