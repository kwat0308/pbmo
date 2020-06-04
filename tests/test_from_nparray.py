'''
Tests performance of C++ vs numpy vs Boost using numpy arrays as input.
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


# test performance benchmark by using a constructed numpy array
# contained of random values
def get_results(rs, cs, args):
    # create numpy array consisting of random values
    arr = np.random.rand(rs, cs)
    # first initialize the matrix in C++
    cpp_mat_t0 = time.time()
    cpp_mat = Matrix.Matrix(arr)
    cpp_mat_t1 = time.time()

    # first initialize the matrix in C++
    boost_mat_t0 = time.time()
    boost_mat = BoostMatrix.BoostMatrix(arr)
    boost_mat_t1 = time.time()

    if args.verbosity > 0:
        cpp_mat.print_mat()  # print c++ matrix
        print(arr)  # print numpy matrix
        boost_mat.print_mat()  # print boost matrix
    if args.verbosity > 3:
        print("Time for creating user-defined matrix ({0}-by-{1}): {2}s\n".
                format(cpp_mat.rows, cpp_mat.cols, cpp_mat_t1 -
                        cpp_mat_t0))  # print time of initialization
        print("Time for creating Boost matrix ({0}-by-{1}): {2}s\n".format(
            boost_mat.rows, boost_mat.cols,
            boost_mat_t1 - boost_mat_t0))  # print time of initialization


    # set number of iterations
    max_iter = 100000

    # now just test performance for evaluation of norm
    result_tup = test_performance(arr, cpp_mat, boost_mat, max_iter, args)

    return result_tup

def test_from_nparray(max_rs, max_cs, args):
    # currently only integrated with square matrix
    # allowing more arbitrary matrices require more work...

    i = 0
    # j = 1
    # max_dim = max_rs*max_cs

    # dictionary that holds results
    # naming convention: type_of_array (where performance is tested)
    # result_dict = {"Dimension":np.zeros(max_dim), "C-array (Python)":np.zeros(max_dim), "NumPy (Python)":np.zeros(max_dim),
    #             "Boost (Python)":np.zeros(max_dim), "C-array (C++)":np.zeros(max_dim), "Boost (C++)":np.zeros(max_dim)}
    result_dict = {"Dimension":np.zeros(max_rs), "C-array (Python)":np.zeros(max_rs), "NumPy (Python)":np.zeros(max_rs),
                "Boost (Python)":np.zeros(max_rs), "C-array (C++)":np.zeros(max_rs), "Boost (C++)":np.zeros(max_rs)}

    for val in np.arange(10, max_rs, 10):
        # results containing performance times for norm evaluation
        result_tup = get_results(val, val, args)
        # append to some data structure
        result_dict["Dimension"][i] = val
        result_dict["C-array (Python)"][i] = result_tup[0]
        result_dict["NumPy (Python)"][i] = result_tup[1]
        result_dict["Boost (Python)"][i] = result_tup[2]
        result_dict["C-array (C++)"][i] = result_tup[3]
        result_dict["Boost (C++)"][i] = result_tup[4]
        # j = i - 1
        i += 1

    # now get plots
    plot_results(result_dict, "nparray", args)


def main():

    # create arg parser
    # create an argument parser
    parser = argparse.ArgumentParser(
        description=
        "Tests performance benchmarks for C++ and Python using numpy arrays."
    )
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
        "Set program to debug mode (verbosity = 4, use max. dimension of 10)"
    )
    # create argument object 
    args = parser.parse_args()

    # first ask user for input for row / columns
    if args.debug_mode:
        max_rs = 10
        max_cs = 10
        args.verbosity = 4
    else:
        # max_rs = int(input("Please enter max. dimension for rows for matrix: "))
        # max_cs = int(input("Please enter max. dimension for columns for matrix: "))
        # max_rs = int(input("Please input max dimension for n-by-n square matrix: "))
        # inputs are hard to implement for plotting, will be implemented in the future
        max_rs = 1000
        max_cs = max_rs
    
    test_from_nparray(max_rs, max_cs, args)
    


main()