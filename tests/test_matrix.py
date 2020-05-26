# This is a test function to test the performance of our compiled Matrix library to numpy arrays

import sys
import os

import time  # for measuring performance
import numpy as np
import Matrix

# sys.path.append("./data")

cwd = os.getcwd()


def make_data(rs, cs, size):

    #small dataset
    np.savetxt(os.path.join(cwd, "data", "{0}_data.txt".format(size)),
               np.random.rand(rs, cs),
               delimiter="\t",
               fmt="%.6f")


def mat(rs, cs, size):

    cpp_mat_t0 = time.time()
    cpp_mat = Matrix.Matrix(rs, cs,
                            "data/{0}_data.txt".format(size))  # 3x4 matrix
    cpp_mat_t1 = time.time()
    cpp_mat.print_mat()  # print c++ matrix
    np_mat_t0 = time.time()
    np_mat = np.loadtxt("data/{0}_data.txt".format(size), delimiter="\t")
    np_mat_t1 = time.time()
    print(np_mat, '\n')
    print("Time for creating C++ matrix ({0}): {1}s".format(
        size, cpp_mat_t1 - cpp_mat_t0))
    print("Time for creating numpy matrix ({0}): {1}s\n".format(
        size, np_mat_t1 - np_mat_t0))

    # now perform norm for both C++ and np matrix

    cpp_norm_t0 = time.time()
    cpp_norm = cpp_mat.norm()
    cpp_norm_t1 = time.time()
    np_norm_t0 = time.time()
    np_norm = np.linalg.norm(np_mat)
    np_norm_t1 = time.time()
    print("Norm values: C++: {0}, numpy: {1}".format(cpp_norm, np_norm))
    print("Time for C++ matrix norm ({0}): {1}s".format(
        size, cpp_norm_t1 - cpp_norm_t0))
    print("Time for numpy matrix norm ({0}): {1}s\n".format(
        size, np_norm_t1 - np_norm_t0))

    # print(np_smat)


def main():

    # parameters for small / large files / matrices
    # small matrix(3x4)
    srs = 3
    scs = 4
    ssize = "small"

    # large matrix (3000x4000)
    lrs = 3000
    lcs = 4000
    lsize = "large"

    size_config = [(srs, scs, ssize), (lrs, lcs, lsize)]

    
    for (rs, cs, size) in size_config:
        fpath = os.path.join(cwd, "data", "{0}_data.txt".format(size))
        # create our text files if not created yet
        if not os.path.isfile(fpath):
            make_data(rs, cs, size)

        # test our matrices
        mat(rs, cs, size)


main()