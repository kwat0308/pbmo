import numba as nb
from numba import cuda
import os
import time
import numpy as np

# required to put in here since jitclass is cumbersome
# and self raises problems with numba JIT compilation
@nb.jit(nopython=True)
def matmul_core(M, N, K, arr1, arr2, prod):
    '''Core of matrix multiplication calculation'''
    for i in range(M):
        for j in range(N):
            for k in range(K):
                prod[i][k] += arr1[i, j] * arr2[j, k]

    return prod

# same as above but uses CPU parallelization
@nb.jit(nopython=True, parallel=True)
def matmul_core_parallel(M, N, K, arr1, arr2, prod):
    '''Core of matrix multiplication calculation'''
    for i in nb.prange(M):
        for j in nb.prange(N):
            for k in nb.prange(K):
                prod[i][k] += arr1[i, j] * arr2[j, k]

    return prod

# cuda version of code, written as CUDA kernel in Python syntax
@cuda.jit
def matmul_core_cuda(M, N, K, arr1, arr2, prod):
    '''Core of matrix multiplication calculation'''
    # index row and column as in threads and blocks in GPUs
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    tmp_sum = 0.
    if row < M and col < N:
        for k in range(K):
            tmp_sum += arr1[row, k] * arr2[k, col]
        prod[row, col] = tmp_sum


class nbMatrix:
    '''
    Naive Python Implementation of Matrix, initialized with NumPy arrays. Contains basic Matrix Operations for
    Performance Evaluations. This matrix is boosted with Numba with JIT compilation.

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

    def norm_core(self):
        '''Core of norm calculation'''
        norm2 = 0.
        for i in range(self.nrows):
            for j in range(self.ncols):
                norm2 += self.arr[i, j] * self.arr[i, j]

        return np.sqrt(norm2)

    def norm(self):
        '''Naive Python implementation of Frobenius Norm'''
        pass

    def matmul(self, mat, return_time=False):
        '''Naive Python implememtation of matrix product'''
        # raise error if ncols dont match with nrows of array
        if self.ncols != mat.nrows:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, mat.nrows))

        prod = np.zeros((self.nrows, mat.ncols))

        # create temporary array to deal with (with compiler) iteration of numba
        tmp_prod = np.zeros((self.nrows, mat.ncols))

        # compile first
        matmul_core(self.nrows, self.ncols, mat.ncols,
                    self.arr, mat.arr, tmp_prod)

        # deallocate memory of temp_prod since we dont need it
        del tmp_prod

        # now evaluate matmul
        # start performance timer
        t0 = time.perf_counter_ns()
        prod = matmul_core(self.nrows, self.ncols,
                           mat.ncols, self.arr, mat.arr, prod)
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)

        return (nbMatrix(prod), eval_time) if return_time else nbMatrix(prod)


class nbparMatrix(nbMatrix):
    '''
    Naive Python Implementation of Matrix, initialized with NumPy arrays. Contains basic Matrix Operations for
    Performance Evaluations. This matrix is boosted with Numba with JIT compilation.
    This matrix utilizes the parallelization scheme provided from numba as well.

    Members
    ------------
    - arr : the matrix, initialized by a NumPy Array
    - nrows : the number of rows (default: 50)
    - ncols : the number of columns (default: 50)
    '''

    def norm_core(self):
        '''Core of norm calculation'''
        norm2 = 0.
        for i in range(self.nrows):
            for j in range(self.ncols):
                norm2 += self.arr[i, j] * self.arr[i, j]

        return np.sqrt(norm2)

    def norm(self):
        '''Naive Python implementation of Frobenius Norm'''
        pass

    def matmul(self, mat, return_time=False):
        '''Naive Python implememtation of matrix product'''
        # raise error if ncols dont match with nrows of array
        if self.ncols != mat.nrows:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, mat.nrows))

        prod = np.zeros((self.nrows, mat.ncols))

        # create temporary array to deal with (with compiler) iteration of numba
        tmp_prod = np.zeros((self.nrows, mat.ncols))

        # compile first
        matmul_core_parallel(self.nrows, self.ncols, mat.ncols,
                             self.arr, mat.arr, tmp_prod)

        # deallocate memory of temp_prod since we dont need it
        del tmp_prod

        # now evaluate matmul
        # start performance timer
        t0 = time.perf_counter_ns()
        prod = matmul_core_parallel(self.nrows, self.ncols,
                                    mat.ncols, self.arr, mat.arr, prod)
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)

        return (nbparMatrix(prod), eval_time) if return_time else nbparMatrix(prod)


class nbcudaMatrix:
    '''
    Naive Python Implementation of Matrix, initialized with NumPy arrays. Contains basic Matrix Operations for
    Performance Evaluations.
    This matrix uses CUDA with Numba with its JIT compilation instead.

    Members
    ------------
    - arr : the matrix, initialized by a NumPy Array
    - nrows : the number of rows (default: 50)
    - ncols : the number of columns (default: 50)
    '''
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

            # solution to grid / block dimension resolving is found from here:
            # https://stackoverflow.com/questions/14504580/pycuda-blocks-and-grids-to-work-with-big-datas
            dx, mx = divmod(self.ncols, self.block_dim[0])
            dy, my = divmod(self.nrows, self.block_dim[1])

            # self.grid_dim = (int(np.ceil(self.nrows /
            #                              self.MAX_THREADS_PER_DIM)),
            #                  int(np.ceil(self.ncols /
            #                              self.MAX_THREADS_PER_DIM)), 1)
            self.grid_dim = (int(dx + (mx > 0)), int(dy + (my > 0)), 1)
        # otherwise set to row and column dimension
        else:
            self.block_dim = (int(self.nrows), int(self.ncols), 1)
            self.grid_dim = (1, 1, 1)

    def allocate_memory(self):
        '''Allocate memory on device and transfer data from host to device'''
        # transfer data from host to device
        self.arr_gpu = cuda.to_device(self.arr)

    def norm_core(self):
        '''Core of norm calculation'''
        norm2 = 0.
        for i in range(self.nrows):
            for j in range(self.ncols):
                norm2 += self.arr[i, j] * self.arr[i, j]

        return np.sqrt(norm2)

    def norm(self):
        '''Naive Python implementation of Frobenius Norm'''
        pass

    def matmul(self, mat, return_time=False):
        '''Naive Python implememtation of matrix product'''
        # raise error if ncols dont match with nrows of array
        if self.ncols != mat.nrows:
            raise ValueError("Dimensions {0} and {1} do not match.".format(
                self.ncols, mat.nrows))

        prod = np.zeros((self.nrows, mat.ncols))
        prod_gpu = cuda.to_device(prod)

        # create temporary array to deal with (with compiler) iteration of numba
        # tmp_prod = np.zeros((self.nrows, mat.ncols))

        # compile first
        # matmul_core_parallel(self.nrows, self.ncols, mat.ncols,
        #                      self.arr, mat.arr, tmp_prod)

        # deallocate memory of temp_prod since we dont need it
        # del tmp_prod

        # now evaluate matmul
        # start performance timer
        t0 = time.perf_counter_ns()
        matmul_core_cuda[self.grid_dim, self.block_dim](
            self.nrows, mat.ncols, self.ncols, self.arr_gpu, mat.arr_gpu, prod_gpu)
        t1 = time.perf_counter_ns()

        eval_time = (t1 - t0) * (1e-9)

        prod = prod_gpu.copy_to_host()

        return (nbparMatrix(prod), eval_time) if return_time else nbparMatrix(prod)
