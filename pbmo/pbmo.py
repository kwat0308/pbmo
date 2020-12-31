import os
import path
import time

from pbmo.lib._libpbmo import Matrix, BoostMatrix
from pbmo.lib.pymatrix import pyMatrix, npMatrix
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(os.path.dirname(ROOT_PATH), "pbmo_plots")


class PBMO:
    '''
    Class that contains the performance benchmarks of the matrix operations from a provided matrix implementation

    Members
    -------
    - dims : the number of dimensions / array of dimensions of matrix / matrices
    - max_iter : the maximum number of iterations for performance evaluation
    - matrix_types : the types of implementations currently available for matrix
    '''

    def __init__(self, dims=(10, 10), max_iter=10000, exclude_matrices=[]):
        '''
        Initialize the performance benchmark evaluator

        Parameters
        ----------
        - dims : tuple of ints or array of int tuples
                The dimension of the initialized matrix / array of dimensions of matrices
        - max_iter : int
                The maximum number of iterations to perform evaluation for
        - exclude_matrices : list of str
                List of matrices to exclude in evaluation
        '''
        self.dims = dims
        self.max_iter = max_iter
        self.exclude_matrices = exclude_matrices

        # reduce matrix types to those which we only want
        self.matrix_types = [
            "Python", "C++", "Boost", "NumPy"
        ]

        # remove matrix types that we dont want
        for mat_type in exclude_matrices:
            self.matrix_types.remove(mat_type)

        self.ntypes = len(self.matrix_types)

        # storage solution incase either are not evaluated
        self.normtimes = 0. if isinstance(self.dims, tuple) else np.zeros(
            (self.ntypes, len(self.dims)))
        self.matmultimes = 0. if isinstance(self.dims, tuple) else np.zeros(
            (self.ntypes, len(self.dims)))

        # results
        # elements are either single value (for single dim evaluation) or have len(self.dims)
        # Note: Ratio is the ratios of average time of mat_type to average time from Python
        self.results = {mat_type: {"Norm Time": 0., "Norm Ratio": 0., "Matmul Time": 0., "Matmul Ratio": 0.}
                        for mat_type in self.matrix_types}
        self.results["Dimensions"] = self.dims

        # headers used for printing results as tabular format
        self.headers = [
            "Type", "Average Norm Evaluation Time [s]", "Norm Ratio to Python",
            "Average Matmul Evaluation Time [s]", "Matmul Ratio to Python"
        ]

    def initialize_matrices(self, arr):
        '''Initialize all matrices that are not excluded from the given list'''

        # initialize matrix and store them in dictionary
        matrix_dict = {
            "Python": pyMatrix(arr),
            "C++": Matrix(arr),
            "Boost": BoostMatrix(arr),
            "Numpy": npMatrix(arr)
        }

        # remove those that are excluded
        # TODO: add some error condition if names dont align
        for matrix_name in self.exclude_matrices:
            matrix_dict.pop(matrix_name)

        return list(matrix_dict.values())

    def evaluate_norm(self, show_progress=False):
        '''
        Evaluate the performance of the norm of all given matrix types
        in all dimensions

        Parameters:
        --------
        - show_progress : bool
            - shows the current dimension it is working on. used for debug purposes
        '''
        # separate between single dimension case vs list of dimensions case
        if isinstance(self.dims, tuple):
            normtimes = self.evaluate_norm_dim(
                self.dims)
        else:
            normtimes = np.zeros((self.ntypes, len(self.dims)))
            # iterate through each dimension
            for k, dim in enumerate(self.dims):
                if show_progress:
                    print("Current Matrix Dimension: {0} x {1}".format(*dim))
                norm_time = self.evaluate_norm_dim(
                    dim)

                normtimes[:, k] = norm_time

        self.normtimes = normtimes

        return normtimes

    def evaluate_norm_dim(self, dim):
        '''Evaluate the norm for matrix with particular dimension'''

        time_arr = np.zeros((self.max_iter, self.ntypes))
        # iterate from i = 0 to N+1
        for i in range(self.max_iter+1):
            # initialize random numpy array and initialize matrices
            # do this for each iteration to make evaluations truly random
            rand_arr = np.array(np.random.rand(
                *dim), copy=False).astype(np.float32)

            matrix_list = self.initialize_matrices(
                rand_arr)

            for j, mat in enumerate(matrix_list):
                t0 = time.perf_counter_ns()
                norm_val = mat.norm()
                t1 = time.perf_counter_ns()

                # append
                # start from index 1 since evaluation @ zero index
                # can have unwanted system effects
                # x 1e-9 to convert to seconds
                time_arr[i-1][j] = (t1 - t0) * (1e-9)

        norm_time = np.mean(time_arr, axis=0)

        return norm_time

    def evaluate_matmul(self):
        '''Evaluate the performance of matrix multiplication (in future release)'''
        pass

    def collect_results(self):
        '''Gather evaluated results and put them into (an) organized dictionary(ies)'''

        for j, mat_type in enumerate(self.matrix_types):
            self.results[mat_type]["Norm Time"] = self.normtimes[j]
            self.results[mat_type]["Norm Ratio"] = self.normtimes[j] / \
                self.normtimes[0]
            self.results[mat_type]["Matmul Time"] = self.matmultimes[j]
            self.results[mat_type]["Matmul Ratio"] = self.matmultimes[j] / \
                self.matmultimes[0]

    def print_results(self):
        '''Print results in tabular format'''
        if isinstance(self.dims, tuple):
            print("Matrix Dimension: {0} x {1}".format(*self.dims))

            table = [
                [mat_type] + list(self.results[mat_type].values()) for mat_type in self.matrix_types
            ]

            print(tabulate(table, headers=self.headers, tablefmt="pretty"))

        else:
            for k, dim in enumerate(self.dims):
                print("Matrix Dimension: {0} x {1}".format(*dim))

                table = []
                for mat_type in self.matrix_types:
                    vals = list(self.results[mat_type].values())
                    val_list = [vals[i][k] for i in range(len(vals))]
                    table.append([mat_type] + val_list)

                print(tabulate(table, headers=self.headers, tablefmt="pretty"))

    # obtained from matplotlib

    def autolabel(ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def plot_results(self, op_type="Norm", plot_ratios=False):
        '''
        Plot performance benchmark results.
        - Plot a bar chart if we only evaluate for single matrix dimension
        - Plot a graph of dimensions vs time for list of dimensions
        '''
        time_arr = np.array(
            [self.results[mat_type]["{0} Time".format(
                op_type)] for mat_type in self.matrix_types]
        )

        ratio_arr = np.array(
            [self.results[mat_type]["{0} Ratio".format(
                op_type)] for mat_type in self.matrix_types]
        )

        # make directory for plots if not already there
        if not os.path.exists(PLOT_PATH):
            os.makedirs(PLOT_PATH)

        if isinstance(self.dims, tuple):
            fig, ax = plt.subplots(figsize=(12, 6))

            locs = np.arange(len(self.matrix_types))  # label locations
            width = 0.35   # bar width

            rects = ax.bar(locs, time_arr, width)
            ax.set_xticks(locs)
            ax.set_xticklabels(self.matrix_types)
            ax.set_ylabels("Average Evaluation Time [s]")
            ax.set_title(
                "Performance Benchmarks for {0} Evaluation with a {1} x {2} Matrix of \
                     Different Implemenations over {3} Iterations".format(op_type, *self.dims, self.max_iter))

            self.autolabel(ax, rects)
            fig.tight_layout()

            plt.savefig(os.path.join(
                PLOT_PATH, "pbmo_{0}_barplot.png".format(op_type)))

        else:
            fig, ax = plt.subplots(figsize=(12, 6))

            for j, mat_type in enumerate(self.matrix_types):
                ax.plot(
                    self.dims, time_arr[j], label=mat_type, marker="o", ms=4.0, lw=2.0)

            ax.set_xlabel("Matrix Dimension")
            ax.set_ylabel("Average Evaluation Time [s]")
            ax.set_title("Performance Benchmarks for {0} Evaluation with Different Implementations over {1} Iterations".format(
                op_type, self.max_iter))

            ax.legend()
            fig.tight_layout()

            plt.savefig(os.path.join(
                PLOT_PATH, "pbmo_{0}_ndims_plot.png".format(op_type)))

            if plot_ratios:
                fig_ratio, ax_ratio = plt.subplots(figsize=(12, 6))

                for j, mat_type in enumerate(self.matrix_types):
                    ax_ratio.plot(
                        self.dims, ratio_arr[j], label=mat_type, marker="o", ms=4.0, lw=2.0)

                ax_ratio.set_xlabel("Matrix Dimension")
                ax_ratio.set_ylabel("Performance Ratio to Python")
                ax_ratio.set_title("Performance Benchmarks for {0} Evaluation with Different Implementations over {1} Iterations".format(
                    op_type, self.max_iter))

                ax_ratio.legend()
                fig_ratio.tight_layout()

                plt.savefig(os.path.join(
                    PLOT_PATH, "pbmo_{0}_ndims_plot_ratio.png".format(op_type)))
