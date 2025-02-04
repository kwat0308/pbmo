import os
import time
import numpy as np
import cupy as cp


class pyMatrix:
    '''
    Naive Python Implementation of Matrix, initialized with NumPy arrays. Contains basic Matrix Operations for
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
        if arr is not None:
            self.arr = arr.astype(np.float32)
            self.nrows = arr.shape[0]
            self.ncols = arr.shape[1]

        else:
            self.nrows = nrows
            self.ncols = ncols
            self.arr = np.zeros((nrows, ncols), dtype=np.float32)

    def norm(self):
        '''Naive Python implementation of Frobenius Norm'''
        norm2 = 0.
        for i in range(self.nrows):
            for j in range(self.ncols):
                norm2 += self.arr[i, j] * self.arr[i, j]

        return np.sqrt(norm2)

    def matmul(self, mat, return_time=False):
        '''Naive Python implememtation of matrix product'''
        # raise error if ncols dont match with nrows of array
        if self.ncols != mat.nrows:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, mat.nrows))

        prod = np.zeros((self.nrows, mat.ncols))

        # start performance timer
        t0 = time.perf_counter_ns()
        for i in range(self.nrows):
            for j in range(self.ncols):
                for k in range(mat.ncols):
                    prod[i][k] += self.arr[i, j] * mat.arr[j, k]
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)

        return (pyMatrix(prod), eval_time) if return_time else pyMatrix(prod)


class npMatrix(pyMatrix):
    '''
    NumPy Implementation of Matrix, initialized with NumPy arrays. Contains basic Matrix Operations for
    Performance Evaluations.

    Members
    ------------
    - arr : the matrix, initialized by a NumPy Array
    - nrows : the number of rows (default: 50)
    - ncols : the number of columns (default: 50)
    '''

    def norm(self):
        '''NumPy Norm'''
        return np.linalg.norm(self.arr)

    def matmul(self, mat, return_time=False):
        '''NumPy Matrix Multiplication'''
        # start performance timer
        t0 = time.perf_counter_ns()
        prod_arr = np.matmul(self.arr, mat.arr)
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)
        return (npMatrix(prod_arr),
                eval_time) if return_time else npMatrix(prod_arr)


class cpMatrix(pyMatrix):
    '''
    CuPy Implementation of Matrix, initialized with CuPy arrays. Contains basic Matrix Operations for
    Performance Evaluations.

    Members
    ------------
    - arr : the matrix, initialized by a CuPy Array
    - nrows : the number of rows (default: 50)
    - ncols : the number of columns (default: 50)
    '''

    def __init__(self, arr=None, nrows=50, ncols=50):
        '''
        Parameters
        -----------
        - nrows : the number of rows (default: 50)
        - ncols : the number of columns (default: 50)
        - arr : array initialized on host device
        '''
        if arr is not None:
            # expensive method, but best way to keep array elements consistent
            # between each matrix implementation
            self.arr = cp.asarray(arr, dtype=np.float32)
            self.nrows = arr.shape[0]
            self.ncols = arr.shape[1]

        else:
            self.nrows = nrows
            self.ncols = ncols
            self.arr = cp.zeros((nrows, ncols), dtype=np.float32)

    def norm(self):
        '''CuPy Norm'''
        return cp.linalg.norm(self.arr)

    def matmul(self, mat, return_time=False):
        '''CuPy Matrix Multiplication. arr is an array initialized on host device'''
        # arr = cp.asarray(mat.arr, dtype=np.float32)
        # start performance timer
        t0 = time.perf_counter_ns()
        prod_arr = cp.matmul(self.arr, mat.arr)
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)
        return (cpMatrix(cp.asnumpy(prod_arr)),
                eval_time) if return_time else cpMatrix(cp.asnumpy(prod_arr))
