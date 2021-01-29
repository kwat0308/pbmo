import os
import time
import numpy as np

import pycuda.autoinit  # automatic cleanup
import pycuda.gpuarray as gpuarray
from skcuda.cublas import cublasCreate, cublasDestroy, cublasSgemm


class cublasMatrix():
    '''
    Matrix implementation using pyCUDA. initialized with NumPy arrays. Contains basic Matrix Operations for
    Performance Evaluations.

    Members
    ------------
    - arr : the matrix, initialized by a NumPy Array
    - nrows : the number of rows (default: 50)
    - ncols : the number of columns (default: 50)
    '''
    def __init__(self, arr=None, nrows=50, ncols=50):
        '''
        Parameters
        -----------
        - nrows : the number of rows (default: 50)
        - ncols : the number of columns (default: 50)
        - arr : optional entry for custom array (default: None)
        '''
        # change type to np.float32 since nvidia only supports single precision
        # should also only be numpy array, otherwise should return error of some sort
        if arr is not None:
            self.arr = arr.astype(np.float32)
            self.nrows = np.int32(arr.shape[0])
            self.ncols = np.int32(arr.shape[1])

        else:
            self.nrows = np.int32(nrows)
            self.ncols = np.int32(ncols)
            self.arr = np.random.rand(nrows, ncols).astype(np.float32)

    def matmul(self, mat, return_time=False):
        '''Matrix multiplication between two matrices'''

        # check dimensions first:
        if self.ncols != mat.nrows:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, mat.nrows))

        # move matrices to gpu
        # somehow we need to transpose this to make it work, not sure why tho
        a_gpu = gpuarray.to_gpu(self.arr.T.copy())
        b_gpu = gpuarray.to_gpu(mat.arr.T.copy())
        c_gpu = gpuarray.to_gpu(
            np.zeros((self.nrows, mat.ncols)).astype(np.float32).T.copy())

        # initialize culas context
        h = cublasCreate()

        # evaluate the matrix multiplication
        # cubals syntax as follows:
        # h = handle, "n" : op(A) = A (for op(A) = A^t, use "t"),
        # then list m, n, k if we have m x k dot k x n
        # then first value is scalar in front of product (set to 1)
        # then give array data followed by row dim of each matrix
        # last value is for adding another matrix, set to 0.
        # also record the time it takes for the evaluation
        t0 = time.perf_counter_ns()
        cublasSgemm(h, "n", "n", self.nrows, mat.ncols, self.ncols,
                    np.float32(1.0), a_gpu.gpudata, self.nrows, b_gpu.gpudata,
                    mat.nrows, np.float32(0.0), c_gpu.gpudata, self.nrows)
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)  # time for each matmul evaluation

        # move from device to host
        prod_arr = c_gpu.get().T

        # free allocated memory for handle
        cublasDestroy(h)

        return (cublasMatrix(prod_arr),
                eval_time) if return_time else cublasMatrix(prod_arr)
