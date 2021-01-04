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

    # kernel that contains CUDA C code
    kernel = """
    __global__ void kernel_norm(float val, const float *a, int rsz, int csz)
    {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rsz && col < csz) {
            for (int j=0; j < csz; ++j) {
                val += a[row * csz + j] * a[row * csz + j];
            }
        }
        val = sqrt(val);
    }

    __global__ void kernel_matmul(float* prod, const float *a, const float *b, int a_rsz, int a_csz, int b_csz)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        float tmp_sum = 0.0f;
        if (row < a_rsz && col < b_csz) {
            for (int k = 0; k < a_csz; ++k) {
                tmp_sum += a[row * a_csz + k] * b[k * b_csz + col];
            }
        }
        prod[row * b_csz + col] = tmp_sum;
    }
    """
    # kernel = """
    # __device__ float kernel_norm(const float *a, int rsz, int csz)
    # {
    #     float val = 0.;
    #     for (int i=0; i < rsz; ++i) {
    #         for (int j=0; j < csz; ++j) {
    #             val += a[i * csz + j] * a[i * csz + j];
    #         }
    #     }
    #     return sqrt(val);
    # }
    # """

    mod = SourceModule(kernel)

    # maximum number of threads available (up to 2-D)
    MAX_THREADS_PER_DIM = 32
    # max number of blocks available
    MAX_BLOCKS_PER_DIM = 65535

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

        self.get_gpu_dims()
        self.allocate_memory()

    def get_gpu_dims(self):
        '''Set GPU block and grid dimensions'''

        # set block dimensions to the max thread per block for each dimension
        # if nrows and ncols > max threads on block per dim
        if self.nrows >= self.MAX_THREADS_PER_DIM and self.ncols >= self.MAX_THREADS_PER_DIM:
            self.block_dim = (self.MAX_THREADS_PER_DIM,
                              self.MAX_THREADS_PER_DIM, 1)

            self.grid_dim = (int(np.ceil(self.nrows / self.MAX_THREADS_PER_DIM)),
                             int(np.ceil(self.ncols / self.MAX_THREADS_PER_DIM)), 1)
        # otherwise set to row and column dimension
        else:
            self.block_dim = (int(self.nrows), int(self.ncols), 1)
            self.grid_dim = (1, 1, 1)

    def allocate_memory(self):
        '''Allocate memory on device and transfer data from host to device'''
        # allocate memory on device
        self.arr_gpu = cuda.mem_alloc(self.arr.nbytes)

        # transfer data from host to device
        cuda.memcpy_htod(self.arr_gpu, self.arr)

    # def norm(self):
    #     '''Frobenius Norm'''

    #     self.allocate_memory()

    #     val_gpu = cuda.mem_alloc(32)
    #     # cuda.memcpy_htod(val_gpu, 0.)

    #     nrm = self.mod.get_function("kernel_norm")
    #     nrm(val_gpu, self.arr_gpu, self.nrows, self.ncols,
    #         block=self.block_dim,
    #         grid=self.grid_dim)

    #     val = 0.
    #     cuda.memcpy_dtoh(val, val_gpu)
    #     return val

    def matmul(self, mat):
        '''Matrix multiplication between two matrices'''

        # self.allocate_memory()

        # check dimensions first:
        if self.ncols != mat.nrows:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, mat.nrows))

        # allocate gpu memory for product yield
        prod_arr = np.zeros((self.nrows, mat.ncols))
        prodarr_gpu = cuda.mem_alloc(prod_arr.nbytes)

        mmul = self.mod.get_function("kernel_matmul")
        mmul(prodarr_gpu, self.arr_gpu, mat.arr_gpu, self.nrows, self.ncols, mat.ncols,
             block=self.block_dim, grid=self.grid_dim)

        # initialize an empty array to store data from device to host
        result_arr = np.empty_like(prod_arr)
        cuda.memcpy_dtoh(result_arr, prodarr_gpu)

        return cuMatrix(result_arr)
