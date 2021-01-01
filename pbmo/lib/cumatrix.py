import os
import time
import numpy as np

import pycuda.autoinit  # automatic cleanup
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class cuMatrix:
    '''
    Matrix implementation using pyCUDA. initialized with NumPy arrays. Contains basic Matrix Operations for
    Performance Evaluations.

    Members
    ------------
    - arr : the matrix, initialized by a NumPy Array
    - nrows : the number of rows (default: 50)
    - ncols : the number of columns (default: 50)
    '''

    def __init__(self, nrows=50, ncols=50, arr=None):
        '''
        Parameters
        -----------
        - nrows : the number of rows (default: 50)
        - ncols : the number of columns (default: 50)
        - arr : optional entry for custom array (default: None)
        '''
        # change type to np.float32 since nvidia only supports single precision
        if arr is not None:
            self.arr = arr.astype(np.float32)
            self.nrows = arr.shape[0]
            self.ncols = arr.shape[1]

        else:
            self.nrows = nrows
            self.ncols = ncols
            self.arr = np.random.rand(nrows, ncols).astype(np.float32)
