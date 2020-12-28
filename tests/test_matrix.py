# This is a test function to test the performance of our compiled Matrix library to numpy arrays

from lib.performance import test_performance
import sys
import os
import argparse

import time  # for measuring performance
import numpy as np
import Matrix
import BoostMatrix

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "lib"))


# sys.path.append("./data")

cwd = os.getcwd()  # current working directory


# create a dataset containing of random values and save to a .tsv file
def make_data(rs, cs, size):

    # small dataset
    np.savetxt(os.path.join(cwd, "data", "{0}_data.tsv".format(size)),
               np.random.rand(rs, cs),
               delimiter="\t")


# test performance benchmark by using a constructed numpy array
# contained of random values
def test_with_nparray(rs, cs, scale, args):
    max_iter = 100000

    # now create small / large np array that contains random values
    small_arr = np.array(np.random.rand(rs, cs), copy=False).astype(np.float64)
    large_arr = np.array(np.random.rand(
        rs * scale, cs * scale), copy=False).astype(np.float64)

    # now test performance benchmarks for each array
    for arr in (small_arr, large_arr):
        # first initialize the matrix in C++
        cpp_mat_t0 = time.time()
        # cpp_mat = Matrix.Matrix(arr.shape[0], arr.shape[1],
        #                         arr)
        cpp_mat = Matrix.Matrix(arr)
        cpp_mat_t1 = time.time()

        # first initialize the matrix in C++
        boost_mat_t0 = time.time()
        # boost_mat = Matrix.Matrix(arr.shape[0], arr.shape[1],
        #                         arr)
        boost_mat = BoostMatrix.BoostMatrix(arr)
        boost_mat_t1 = time.time()

        if args.verbosity > 0:
            cpp_mat.print_mat()  # print c++ matrix
            print(arr)  # print numpy matrix
            boost_mat.print_mat()  # print boost matrix
        if args.verbosity > 1:
            print("Time for creating user-defined matrix ({0}-by-{1}): {2}s\n".
                  format(cpp_mat.rows, cpp_mat.cols, cpp_mat_t1 -
                         cpp_mat_t0))  # print time of initialization
            print("Time for creating Boost matrix ({0}-by-{1}): {2}s\n".format(
                boost_mat.rows, boost_mat.cols,
                boost_mat_t1 - boost_mat_t0))  # print time of initialization

        # now just test performance for evaluation of norm
        test_performance(arr, cpp_mat, boost_mat, max_iter, args)


# Test performance benchmarks using data files
# that are constructed in this function
def test_with_datafile(rs, cs, scale, args):

    max_iter = 100000

    size_config = [(rs, cs, "small"), (rs * scale, cs * scale, "large")]

    for (rs, cs, size) in size_config:
        fname = "{0}_data.tsv".format(size)
        fpath = os.path.join(cwd, "data", fname)
        # clear file contents, then construct new datafile with set dimensions
        if os.path.exists(fpath) and os.path.isfile(fpath):
            os.remove(fpath)
        make_data(rs, cs, size)

        # initialize C++ matrix
        cpp_mat_t0 = time.time()
        cpp_mat = Matrix.Matrix(rs, cs, "data/{0}_data.tsv".format(size))
        cpp_mat_t1 = time.time()

        # also initialize numpy array
        np_mat_t0 = time.time()
        np_mat = np.loadtxt("data/{0}_data.tsv".format(size), delimiter="\t")
        np_mat_t1 = time.time()

        # also initialize Boost matrix
        boost_mat_t0 = time.time()
        boost_mat = BoostMatrix.BoostMatrix(rs, cs,
                                            "data/{0}_data.tsv".format(size))
        boost_mat_t1 = time.time()

        if args.verbosity > 1:
            cpp_mat.print_mat()  # print c++ matrix
            print(np_mat, '\n')  # print numpy matrix
            boost_mat.print_mat()  # print boost matrix
        if args.verbosity > 0:
            print("Time for creating user-defined matrix ({0}-by-{1}): {2}s\n".
                  format(cpp_mat.rows, cpp_mat.cols, cpp_mat_t1 -
                         cpp_mat_t0))  # print time of initialization
            print("Time for creating numpy matrix ({0}-by-{1}): {2}s\n".format(
                np_mat.shape[0], np_mat.shape[1],
                np_mat_t1 - np_mat_t0))  # print time of initialization
            print("Time for creating Bopst matrix ({0}-by-{1}): {2}s\n".format(
                boost_mat.rows, boost_mat.cols,
                boost_mat_t1 - boost_mat_t0))  # print time of initialization

        test_performance(np_mat, cpp_mat, boost_mat, max_iter, args)


def main():

    # create arg parser
    # create an argument parser
    parser = argparse.ArgumentParser(
        description="Tests performance benchmarks for C++ and Python using data files and numpy arrays."
    )
    # add necessary flags
    # choose whether to test performance benchmark with datafile or with numpy arrays or both
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        type=str,
        default="both",
        help="Choose test mode (datafile (using datafiles), np_array(using numpy arrays), or both)."
    )
    # set verbosity
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="verbosity",
        type=int,
        default=0,
        help="Set the level of verbosity for debugging purposes (0 (lowest) to 4 (highest))."
    )
    # set debug mode
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug_mode",
        action="store_true",
        help="Set program to debug mode (verbosity = 4, use default row / column size of 3, 4)"
    )
    # create argument object that contains truth conditions for plot condition or save condition
    args = parser.parse_args()
    mode = args.mode

    # first ask user for input for row / columns
    if args.debug_mode:
        rs = 3
        cs = 4
        scale = 50  # set scaling factor by 50 for large matrices
        args.verbosity = 4
    else:
        rs = int(input("Please enter dimension for rows for matrix: "))
        cs = int(input("Please enter dimension for columns for matrix: "))
        scale = int(
            input(
                "Enter the scaling factor (integer) used for performance test for a large matrix: "
            ))

    if mode.find("np_array") != -1:
        test_with_nparray(rs, cs, scale, args)  # test with numpy array
    elif mode.find("datafile") != -1:
        test_with_datafile(rs, cs, scale, args)  # test with datafiles
    elif mode.find("both") != -1:
        test_with_nparray(rs, cs, scale, args)
        test_with_datafile(rs, cs, scale, args)
    else:
        raise Exception(
            "Invalid mode type, choose from datafile, np_array, or both.")


main()
'''

# test the performance of the evaluation of C++ and numpy norm
def test_performance(arr, cpp_mat, boost_mat, args):
    max_iter = 10000  # number of iterations for evaluation of norm
    dim = arr.shape

    # get average time of norm evaluation for both arrays

    cpp_normarr = np.zeros(max_iter)
    np_normarr = np.zeros(max_iter)
    boost_normarr = np.zeros(max_iter)

    cpp_timearr = np.zeros(max_iter)
    np_timearr = np.zeros(max_iter)
    boost_timearr = np.zeros(max_iter)

    i = 0
    while (i <= max_iter):
        #user-defined norm
        cpp_norm_t0 = time.time()
        cpp_norm = cpp_mat.norm()
        cpp_norm_t1 = time.time()
        # numpy linalg norm
        np_norm_t0 = time.time()
        np_norm = np.linalg.norm(arr)
        np_norm_t1 = time.time()
        # boost norm
        boost_norm_t0 = time.time()
        boost_norm = boost_mat.norm()
        boost_norm_t1 = time.time()

        # progression status message for each tenth of the loop
        if i % (max_iter // 10) == 0:
            print("Progression status: {0} iterations completed.\n".format(i))
            if args.verbosity > 3:
                print("Results for {0}th iteration:\n".format(i))
                print("Norm values: user-defined: {0}, numpy: {1}, boost: {2}".
                      format(cpp_norm, np_norm, boost_norm))
                print(
                    "Time for user-defined matrix norm ({0}-by-{1} matrix): {2}s"
                    .format(dim[0], dim[1], cpp_norm_t1 - cpp_norm_t0))
                print("Time for numpy matrix norm ({0}-by-{1} matrix): {2}s".
                      format(dim[0], dim[1], np_norm_t1 - np_norm_t0))
                print("Time for boost norm ({0}-by-{1} matrix): {2}s\n".format(
                    dim[0], dim[1], boost_norm_t1 - boost_norm_t0))

        # only append after first iteration
        if i > 0:
            cpp_normarr[i - 1] = cpp_norm
            np_normarr[i - 1] = np_norm
            boost_normarr[i - 1] = boost_norm

            cpp_timearr[i - 1] = cpp_norm_t1 - cpp_norm_t0
            np_timearr[i - 1] = np_norm_t1 - np_norm_t0
            boost_timearr[i - 1] = boost_norm_t1 - boost_norm_t0

        i += 1  # increment

    # get the average values of the norm and times
    cpp_avgnorm = np.mean(cpp_normarr)
    np_avgnorm = np.mean(np_normarr)
    boost_avgnorm = np.mean(boost_normarr)

    cpp_avgtime = np.mean(cpp_timearr)
    np_avgtime = np.mean(np_timearr)
    boost_avgtime = np.mean(boost_timearr)

    # get results from evaluating performance from c++
    # i.e. for loop within c++
    cpp_results_cpp = cpp_mat.norm_performance(max_iter)
    boost_results_cpp = boost_mat.norm_performance(max_iter)

    # print performance results
    print_results(cpp_avgnorm, np_avgnorm, boost_avgnorm, cpp_results_cpp,
                  boost_results_cpp, cpp_avgtime, np_avgtime, boost_avgtime,
                  dim, max_iter, args)


# print results of our performance benchmarks
def print_results(cpp_norm, np_norm, b_norm, cppresults_cpp, bresults_cpp,
                  cpp_time, np_time, b_time, dim, max_iter, args):
    print(
        "Results for performance benchmarking for {0}-by-{1} matrix:\n".format(
            *dim),
        "Number of iterations for performance benchmark: {0}\n".format(
            max_iter))
    if args.verbosity > 1:
        print(
            "Average norm evaluated from user-defined class: {0},\nAverage norm evaluated from numpy: {1},\nAverage norm evaluated from Boost: {2}\n"
            .format(cpp_norm, np_norm, b_norm)),
    print(
        "Average time taken for evaluation of norm from user-defined class: {0} s\n"
        .format(cpp_time),
        "Average time taken for evaluation of norm from numpy: {0} s\n".format(
            np_time),
        "Average time taken for evaluation of norm from Boost: {0} s\n".format(
            b_time),
        "Performance from evaluating from C++ (user-defined matrix): value: {0}, time: {1}\n"
        .format(*cppresults_cpp),
        "Performance from evaluating from C++ (boost matrix): value: {0}, time: {1}\n"
        .format(*bresults_cpp))
    if args.verbosity > 1:
        print(
            "\nComparisons:\n",
            "Accuracy of user-defined norm vs numpy norm: {0}\n".format(
                np.abs(np_norm - cpp_norm)),
            "Accuracy of user-defined norm vs Boost norm: {0}\n".format(
                np.abs(cpp_norm - b_norm)),
            "Accuracy of numpy norm vs Boost norm: {0}\n".format(
                np.abs(np_norm - b_norm)),
            "Ratio of user-defined matrix performance to numpy performance: {0}\n"
            .format(cpp_time / np_time),
            "Ratio of Boost performance to numpy performance: {0}\n".format(
                b_time / np_time),
            "Ratio of user-defined matrix performance to Boost performance: {0}\n"
            .format(cpp_time / b_time))



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
