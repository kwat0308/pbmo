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
            self.arr = arr
            self.nrows = arr.shape[0]
            self.ncols = arr.shape[1]

        else:
            self.nrows = nrows
            self.ncols = ncols
            self.arr = np.zeros((nrows, ncols))

    def norm(self):
        '''Naive Python implementation of Frobenius Norm'''
        norm2 = 0.
        for i in range(self.nrows):
            for j in range(self.ncols):
                norm2 += self.arr[i, j] * self.arr[i, j]

        return np.sqrt(norm2)

    def matmul(self, arr):
        '''Naive Python implememtation of matrix product'''
        # raise error if ncols dont match with nrows of array
        if self.ncols != arr.shape[0]:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, arr.shape[0]))

        prod = np.zeros((self.ncols, arr.shape[0]))
        for i in range(self.nrows):
            for j in range(self.ncols):
                for k in range(arr.shape[1]):
                    prod[i][j] = self.arr[i, j] * arr[j, k]

        return prod


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

    def matmul(self, arr):
        '''NumPy Matrix Multiplication'''
        return np.matmul(self.arr, arr)


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

    def __init__(self, nrows=50, ncols=50, arr=None):
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
            self.arr = cp.zeros((nrows, ncols))

    def norm(self):
        '''CuPy Norm'''
        return cp.linalg.norm(self.arr)

    def matmul(self, arr):
        '''CuPy Matrix Multiplication. arr is an array initialized on host device'''
        arr = cp.asarray(arr, dtype=np.float32)
        return cp.matmul(self.arr, arr)
