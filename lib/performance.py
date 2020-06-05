'''
Python library that contains functions used for performance benchmarking.
'''
import os, time
import Matrix, BoostMatrix
import numpy as np
import matplotlib.pyplot as plt


# evaluate a norm in a naive Python loop
# arr should be a numpy array with some given shape
def eval_norm(arr):
    norm = 0.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            norm += arr[i, j] * arr[i, j]

    return np.sqrt(norm)


# print results of our performance benchmarks
def print_results(py_norm, cpp_norm, np_norm, b_norm, cppresults_cpp,
                  bresults_cpp, py_time, cpp_time, np_time, b_time, dim,
                  max_iter, args):
    print(
        "Results for performance benchmarking for {0}-by-{1} matrix:\n".format(
            *dim),
        "Number of iterations for performance benchmark: {0}\n".format(
            max_iter))
    if args.verbosity > 1:
        print(
            "Average norm evaluated from Python: {0},\nAverage norm evaluated from C++: {1},\nAverage norm evaluated from numpy: {2}\nAverage norm evaluated from Boost: {3}"
            .format(py_norm, cpp_norm, np_norm, b_norm)),
    print(
        "Average time taken for evaluation of norm from Python: {0} s\n".
        format(py_time),
        "Average time taken for evaluation of norm from C++: {0} s\n".format(
            cpp_time),
        "Average time taken for evaluation of norm from numpy: {0} s\n".format(
            np_time),
        "Average time taken for evaluation of norm from Boost: {0} s\n".format(
            b_time),
    )
    # "Performance from evaluating from C++ (user-defined matrix): value: {0}, time: {1}\n"
    # .format(*cppresults_cpp),
    # "Performance from evaluating from C++ (boost matrix): value: {0}, time: {1}\n"
    # .format(*bresults_cpp))
    if args.verbosity > 1:
        print(
            "\nComparisons:\n",
            "Accuracy of C++ norm vs Python norm: {0}\n".format(
                np.abs(np_norm - py_norm)),
            "Accuracy of C++ norm vs numpy norm: {0}\n".format(
                np.abs(np_norm - cpp_norm)),
            "Accuracy of C++ norm vs Boost norm: {0}\n".format(
                np.abs(cpp_norm - b_norm)),
            # "Accuracy of numpy norm vs Boost norm: {0}\n".format(
            #     np.abs(np_norm - b_norm)),
            "Ratio of C++ matrix performance to Python performance: {0}\n".
            format(cpp_time / py_time),
            "Ratio of C++ matrix performance to numpy performance: {0}\n".
            format(cpp_time / np_time),
            # "Ratio of Boost performance to numpy performance: {0}\n".format(
            #     b_time / np_time),
            "Ratio of C++ matrix performance to Boost performance: {0}\n".
            format(cpp_time / b_time))


# test the performance of the evaluation of C++ and numpy norm
# returns tuple that contains results
# should figure out better way to get results in some data structure... (maybe dict?)
def test_performance(arr, cpp_mat, boost_mat, max_iter, args):
    dim = arr.shape

    # get average time of norm evaluation for both arrays

    py_normarr = np.zeros(max_iter)
    cpp_normarr = np.zeros(max_iter)
    np_normarr = np.zeros(max_iter)
    boost_normarr = np.zeros(max_iter)

    py_timearr = np.zeros(max_iter)
    cpp_timearr = np.zeros(max_iter)
    np_timearr = np.zeros(max_iter)
    boost_timearr = np.zeros(max_iter)

    i = 0
    while (i <= max_iter):
        # naive python norm
        py_norm_t0 = time.time()
        py_norm = eval_norm(arr)
        py_norm_t1 = time.time()
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
                print(
                    "Norm values: Python: {0}, C++: {1}, numpy: {2}, boost: {3}"
                    .format(py_norm, cpp_norm, np_norm, boost_norm))
                print("Time for Python matrix norm ({0}-by-{1} matrix): {2}s".
                      format(dim[0], dim[1], py_norm_t1 - py_norm_t0))
                print("Time for C++ matrix norm ({0}-by-{1} matrix): {2}s".
                      format(dim[0], dim[1], cpp_norm_t1 - cpp_norm_t0))
                print("Time for numpy matrix norm ({0}-by-{1} matrix): {2}s".
                      format(dim[0], dim[1], np_norm_t1 - np_norm_t0))
                print("Time for boost norm ({0}-by-{1} matrix): {2}s\n".format(
                    dim[0], dim[1], boost_norm_t1 - boost_norm_t0))

        # only append after first iteration
        if i > 0:
            py_normarr[i - 1] = py_norm
            cpp_normarr[i - 1] = cpp_norm
            np_normarr[i - 1] = np_norm
            boost_normarr[i - 1] = boost_norm

            py_timearr[i - 1] = py_norm_t1 - py_norm_t0
            cpp_timearr[i - 1] = cpp_norm_t1 - cpp_norm_t0
            np_timearr[i - 1] = np_norm_t1 - np_norm_t0
            boost_timearr[i - 1] = boost_norm_t1 - boost_norm_t0

        i += 1  # increment

    # get the average values of the norm and times
    py_avgnorm = np.mean(py_normarr)
    cpp_avgnorm = np.mean(cpp_normarr)
    np_avgnorm = np.mean(np_normarr)
    boost_avgnorm = np.mean(boost_normarr)

    py_avgtime = np.mean(py_timearr)
    cpp_avgtime = np.mean(cpp_timearr)
    np_avgtime = np.mean(np_timearr)
    boost_avgtime = np.mean(boost_timearr)

    # get results from evaluating performance from c++
    # i.e. for loop within c++
    # returns tuple of (value, time)
    cpp_results_cpp = cpp_mat.norm_performance(max_iter)
    boost_results_cpp = boost_mat.norm_performance(max_iter)

    # print performance results
    if args.verbosity > 0:
        print_results(py_avgnorm, cpp_avgnorm, np_avgnorm, boost_avgnorm,
                      cpp_results_cpp, boost_results_cpp, py_avgtime,
                      cpp_avgtime, np_avgtime, boost_avgtime, dim, max_iter,
                      args)

    # return tuple of times (no need for actual values)
    return (py_avgtime, cpp_avgtime, np_avgtime, boost_avgtime,
            cpp_results_cpp[1], boost_results_cpp[1])


# plot results of performance benchmarks
def plot_results(result_dict, mode, args):
    dim_arr = result_dict.pop("Dimension")
    result_arr = list(result_dict.values())[0:4]
    label_arr = list(result_dict.keys())[0:4]
    color_arr = ['b', 'k', 'g', 'c', 'm']
    fig, ax = plt.subplots(figsize=(12, 9))

    for i in range(len(label_arr)):
        ax.plot(dim_arr,
                result_arr[i],
                label=label_arr[i].replace(" (Python)", ""),
                color=color_arr[i],
                marker='o',
                ms=3)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Evaluation time [s]")
    ax.set_title("Performance Benchmark of Square Matrix Norm Evaluation")
    ax.legend(loc='upper left')

    plt.savefig(os.path.join(os.getcwd(), "data",
                             "result_plot_{:s}.pdf".format(mode)),
                dpi=800)

    if args.verbosity > 2:
        plt.show()
