'''
Tests performance of C++ vs numpy vs Boost using data files as input.
This eliminates unnecessary flags when testing.
'''
''' Creates plots that evaluate the performance of norm evaulation with different matrix dimension. This compares:
    - user-defined C++ (pointers)
    - Boost matrix
    - numpy array
    - pointer and Boost matrix evaluated from C++ loop (separate plot (maybe?))

    This should be able to do this using numpy arrays or files from data as input.

    This uses elements from test_matrix.py.
'''
import os, argparse
import time
import Matrix, BoostMatrix

import numpy as np

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "lib"))

from lib.performance import test_performance, plot_results


# Test performance benchmarks using data files
# that are constructed in this function
def get_results(rs, cs, args):
    fpath = os.path.join(os.getcwd(), "data", "test_data.tsv")
    # remove file if its still there
    if os.path.isfile(fpath):
        os.remove(fpath)
    # construct new datafile with set dimensions
    np.savetxt(fpath,
               np.array(np.random.rand(rs, cs)).astype(np.float64),
               delimiter="\t")

    # initialize C++ matrix
    cpp_mat_t0 = time.time()
    cpp_mat = Matrix.Matrix(rs, cs, "data/test_data.tsv")
    cpp_mat_t1 = time.time()

    # also initialize numpy array
    np_mat_t0 = time.time()
    np_mat = np.loadtxt("data/test_data.tsv", delimiter="\t")
    np_mat_t1 = time.time()

    # also initialize Boost matrix
    boost_mat_t0 = time.time()
    boost_mat = BoostMatrix.BoostMatrix(rs, cs, "data/test_data.tsv")
    boost_mat_t1 = time.time()

    if args.verbosity > 1:
        cpp_mat.print_mat()  # print c++ matrix
        print(np_mat, '\n')  # print numpy matrix
        boost_mat.print_mat()  # print boost matrix
    if args.verbosity > 3:
        print("Time for creating user-defined matrix ({0}-by-{1}): {2}s\n".
              format(cpp_mat.rows, cpp_mat.cols,
                     cpp_mat_t1 - cpp_mat_t0))  # print time of initialization
        print("Time for creating numpy matrix ({0}-by-{1}): {2}s\n".format(
            np_mat.shape[0], np_mat.shape[1],
            np_mat_t1 - np_mat_t0))  # print time of initialization
        print("Time for creating Boost matrix ({0}-by-{1}): {2}s\n".format(
            boost_mat.rows, boost_mat.cols,
            boost_mat_t1 - boost_mat_t0))  # print time of initialization

    # set number of iterations
    max_iter = 100000
    result_tup = test_performance(np_mat, cpp_mat, boost_mat, max_iter, args)

    # clear file for next iteration
    os.remove(fpath)

    return result_tup


def test_from_datafile(max_rs, max_cs, args):
    # currently only integrated with square matrix
    # allowing more arbitrary matrices require more work...

    # start from 2x2 matrix (1x1 makes numpy arrays a vector)
    # i = 0
    # j = 1
    # max_dim = max_rs*max_cs
    arr_len = 10

    # dictionary that holds results
    # naming convention: type_of_array (where performance is tested)
    # result_dict = {"Dimension":np.zeros(max_dim), "C-array (Python)":np.zeros(max_dim), "NumPy (Python)":np.zeros(max_dim),
    #             "Boost (Python)":np.zeros(max_dim), "C-array (C++)":np.zeros(max_dim), "Boost (C++)":np.zeros(max_dim)}
    result_dict = {
        "Dimension": np.zeros(arr_len),
        "Python (Python)": np.zeros(arr_len),
        "C-array (Python)": np.zeros(arr_len),
        "NumPy (Python)": np.zeros(arr_len),
        "Boost (Python)": np.zeros(arr_len),
        "C-array (C++)": np.zeros(arr_len),
        "Boost (C++)": np.zeros(arr_len)
    }

    for i, dim in enumerate(np.linspace(5, max_rs, arr_len, dtype=int)):
        print("Current Matrix dimension: {0}\n".format(dim))
        # results containing performance times for norm evaluation
        result_tup = get_results(dim, dim, args)
        # append to some data structure
        result_dict["Dimension"][i] = dim
        result_dict["Python (Python)"][i] = result_tup[0]
        result_dict["C-array (Python)"][i] = result_tup[1]
        result_dict["NumPy (Python)"][i] = result_tup[2]
        result_dict["Boost (Python)"][i] = result_tup[3]
        result_dict["C-array (C++)"][i] = result_tup[4]
        result_dict["Boost (C++)"][i] = result_tup[5]
        # i += 1
        # j = i - 1

    # now get plots
    plot_results(result_dict, "datafile", args)


def main():

    # create arg parser
    # create an argument parser
    parser = argparse.ArgumentParser(
        description=
        "Tests performance benchmarks for C++ and Python using data files.")
    # add necessary flags
    # set verbosity
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="verbosity",
        type=int,
        default=0,
        help=
        "Set the level of verbosity for debugging purposes (0 (lowest) to 4 (highest))."
    )
    # set debug mode
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug_mode",
        action="store_true",
        help=
        "Set program to debug mode (verbosity = 4, use max. dimension of 10)")
    # create argument object
    args = parser.parse_args()

    # first ask user for input for row / columns
    if args.debug_mode:
        max_rs = 50
        max_cs = 50
        args.verbosity = 4
    else:
        # max_rs = int(input("Please enter max. dimension for rows for matrix: "))
        # max_cs = int(input("Please enter max. dimension for columns for matrix: "))
        # max_rs = int(input("Please input max dimension for n-by-n square matrix: "))

        # inputs are hard to implement for plotting, will be implemented in the future
        max_rs = 500
        max_cs = max_rs

    test_from_datafile(max_rs, max_cs, args)


main()