# This is a test function to test the performance of our compiled Matrix library to numpy arrays

import sys
import os
import argparse

import time  # for measuring performance
import numpy as np
import Matrix

# sys.path.append("./data")

cwd = os.getcwd()   # current working directory

# create a dataset containing of random values and save to a .tsv file
def make_data(rs, cs, size):

    #small dataset
    np.savetxt(os.path.join(cwd, "data", "{0}_data.tsv".format(size)),
               np.random.rand(rs, cs),
               delimiter="\t")

# test the performance of the evaluation of C++ and numpy norm
def test_performance(arr, cpp_mat, args):
    max_iter = 1e6    # number of iterations for evaluation of norm
    dim = arr.shape

    # get average time of norm evaluation for both arrays

    cpp_normarr = np.zeros(max_iter)
    np_normarr = np.zeros(max_iter)

    cpp_timearr = np.zeros(max_iter)
    np_timearr = np.zeros(max_iter)
    
    i=0
    while (i < max_iter):
        #C++ norm
        cpp_norm_t0 = time.time()
        cpp_norm = cpp_mat.norm()
        cpp_norm_t1 = time.time()
        # numpy linalg norm
        np_norm_t0 = time.time()
        np_norm = np.linalg.norm(arr)
        np_norm_t1 = time.time()

        if args.verbosity > 3:
            print("Norm values: C++: {0}, numpy: {1}".format(cpp_norm, np_norm))
            print("Time for C++ matrix norm ({0}-by-{1} matrix): {2}s".format(
                dim[0], dim[1], cpp_norm_t1 - cpp_norm_t0))
            print("Time for numpy matrix norm ({0}-by-{1} matrix): {2}s\n".format(
                dim[0], dim[1], np_norm_t1 - np_norm_t0))

        # only append after first iteration
        if i > 0:
            cpp_normarr[i] = cpp_norm
            np_normarr[i] = np_norm

            cpp_timearr[i] = cpp_norm_t1 - cpp_norm_t0
            np_timearr[i] = np_norm_t1 - np_norm_t0

        i+=1  # increment
    
    # get the average values of the norm and times
    cpp_avgnorm = np.mean(cpp_normarr)
    np_avgnorm = np.mean(np_normarr)

    cpp_avgtime = np.mean(cpp_timearr)
    np_avgtime = np.mean(np_timearr)

    # print performance results
    print_results(cpp_avgnorm, np_avgnorm, cpp_avgtime, np_avgtime, dim, max_iter)


# print results of our performance benchmarks
def print_results(cpp_norm, np_norm, cpp_time, np_time, dim, max_iter):
    print("Results for performance benchmarking for {0}-by-{1} matrix:\n".format(*dim),
    "Number of iterations for performance benchmark: {0}\n".format(max_iter),
    "Average norm evaluated from C++: {0}, \t Average norm evaluated from numpy: {1}\n".format(cpp_norm, np_norm),
    "Average time taken for evaluation of norm from C++: {0} s\n".format(cpp_time),
    "Average time taken for evaluation of norm from numpy: {0} s\n".format(np_time),
    "Comparisons:\n",
    "Accuracy of C++ norm vs numpy norm: {0}\n".format(np.abs(np_norm - cpp_norm)),
    "Ratio of C++ performance to numpy performance: {0}".format(cpp_time / np_time)
    )


# test performance benchmark by using a constructed numpy array
# contained of random values
def test_with_nparray(rs, cs, scale, args):

    # now create small / large np array that contains random values 
    small_arr = np.random.rand(rs, cs)
    large_arr = np.random.rand(rs*scale, cs*scale)

    # now test performance benchmarks for each array
    for arr in (small_arr, large_arr):
        # first initialize the matrix in C++
        cpp_mat_t0 = time.time()
        cpp_mat = Matrix.Matrix(arr.shape[0], arr.shape[1],
                                arr.data)  # 3x4 matrix
        cpp_mat_t1 = time.time()

        if args.verbosity > 1:
            cpp_mat.print_mat()  # print c++ matrix
            print(arr)   # print numpy matrix
        if args.verbosity > 0:
            print("Time for creating C++ matrix ({0}-by-{1}): {2}s\n".format(
                cpp_mat.rowsize(), cpp_mat.columnsize(), cpp_mat_t1 - cpp_mat_t0))   # print time of initialization

        # now just test performance for evaluation of norm
        test_performance(arr, cpp_mat, args)



# Test performance benchmarks using data files 
# that are constructed in this function
def test_with_datafile(rs, cs, scale, args):

    size_config = [(rs, cs, "small"), (rs*scale, cs*scale, "large")]

    for (rs, cs, size) in size_config:
        fname = "{0}_data.txt".format(size)
        fpath = os.path.join(cwd, "data", fname)
        # create our text files if not created yet
        if not os.path.isfile(fpath):
            make_data(rs, cs, size)

        # initialize C++ matrix
        cpp_mat_t0 = time.time()
        cpp_mat = Matrix.Matrix(rs, cs,
                                "data/{0}_data.txt".format(size))  # 3x4 matrix
        cpp_mat_t1 = time.time()
        
        # also initialize numpy array
        np_mat_t0 = time.time()
        np_mat = np.loadtxt("data/{0}_data.txt".format(size), delimiter="\t")
        np_mat_t1 = time.time()

        if args.verbosity > 1:
            cpp_mat.print_mat()  # print c++ matrix
            print(np_mat)   # print numpy matrix
        if args.verbosity > 0:
            print("Time for creating C++ matrix ({0}-by-{1}): {2}s\n".format(
                cpp_mat.rowsize(), cpp_mat.columnsize(), cpp_mat_t1 - cpp_mat_t0))   # print time of initialization
            print("Time for creating numpy matrix ({0}-by-{1}): {2}s\n".format(
                np_mat.shape[0], np_mat.shape[1], np_mat_t1 - np_mat_t0))   # print time of initialization

        test_performance(np_mat, cpp_mat, args)
        


def main():

    # create arg parser
    # create an argument parser
    parser = argparse.ArgumentParser(description="Tests performance benchmarks for C++ and Python using data files and numpy arrays.")
    # add necessary flags
    # choose whether to test performance benchmark with datafile or with numpy arrays or both
    parser.add_argument("-m", "--mode", dest="mode", type=str, default="both", help='Choose test mode (datafile (using datafiles), np_array(using numpy arrays), or both).')
    # set verbosity
    parser.add_argument("-v", "--verbosity", dest="verbosity", type=int, default=0, help="Set the level of verbosity for debugging purposes (0 (lowest) to 4 (highest)).")
    # set debug mode
    parser.add_argument("-d", "--debug", dest="debug_mode", type=bool, default=False, help="Set program to debug mode (verbosity = 3, use default row / column size of 3, 4)")
    # create argument object that contains truth conditions for plot condition or save condition
    args = parser.parse_args()
    mode = args.mode

    # first ask user for input for row / columns
    if args.debug_mode:
        rs = 3
        cs = 4
        args.verbosity = 3
    else:
        rs = int(input("Please enter dimension for rows for matrix: "))
        cs = int(input("Please enter dimension for columns for matrix: "))

    scale = 1000   # set scaling factor by 1000 for large matrices (we can make this user input as well)

    if mode.find("np_array") != -1:
        test_with_nparray(rs, cs, scale, args)    # test with numpy array
    elif mode.find("datafile") != -1:
        test_with_datafile(rs, cs, scale, args)   # test with datafiles
    elif mode.find("both") != -1:
        test_with_nparray(rs, cs, scale, args) 
        test_with_datafile(rs, cs, scale, args)
    else:
        raise Exception("Invalid mode type, choose from datafile, np_array, or both.")

main()



'''
# from previous main function

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

        test_arr = np.random.rand(rs, cs)

        # test our matrices
        mat(rs, cs, size)


# def mat(rs, cs, size):

#     cpp_mat_t0 = time.time()
#     cpp_mat = Matrix.Matrix(rs, cs,
#                             "data/{0}_data.txt".format(size))  # 3x4 matrix
#     cpp_mat_t1 = time.time()
#     cpp_mat.print_mat()  # print c++ matrix
#     np_mat_t0 = time.time()
#     np_mat = np.loadtxt("data/{0}_data.txt".format(size), delimiter="\t")
#     np_mat_t1 = time.time()
#     print(np_mat, '\n')
#     print("Time for creating C++ matrix ({0}): {1}s".format(
#         size, cpp_mat_t1 - cpp_mat_t0))
#     print("Time for creating numpy matrix ({0}): {1}s\n".format(
#         size, np_mat_t1 - np_mat_t0))

#     # now perform norm for both C++ and np matrix

#     cpp_norm_t0 = time.time()
#     cpp_norm = cpp_mat.norm()
#     cpp_norm_t1 = time.time()
#     np_norm_t0 = time.time()
#     np_norm = np.linalg.norm(np_mat)
#     np_norm_t1 = time.time()
#     print("Norm values: C++: {0}, numpy: {1}".format(cpp_norm, np_norm))
#     print("Time for C++ matrix norm ({0}): {1}s".format(
#         size, cpp_norm_t1 - cpp_norm_t0))
#     print("Time for numpy matrix norm ({0}): {1}s\n".format(
#         size, np_norm_t1 - np_norm_t0))

    # print(np_smat)


'''