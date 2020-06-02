#include <stdio.h>
#include <fstream>
#include <string>
#include <time.h>
#include "Matrix.h"

// boost directives
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

// // pybind11 directives for numpy arrays
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// current issue: cannot find <Python.h>, so we need to put this in -I flag
// This will be implemented in the Makefile

using namespace boost::numeric;

void read_from_file(ublas::matrix<double> &bmat, const std::string &fname)
{
    std::ifstream ifs{fname};

    if (!ifs)
    {
        throw std::runtime_error("Cannot open file!");
    }

    double val;

    std::string line;
    int i = 0; // row index

    while (std::getline(ifs, line))
    {                                // read each line in file
        int j = 0;                   // column index
        std::istringstream ss{line}; // stringstream of each row
        while (ss >> val)
        {
            bmat(i, j) = val; // set value
            ++j;              // increment column index
        }

        ++i; // increment row index
    }
}

// void read_from_nparray(ublas::matrix<double> &bmat, const pybind11::array_t<double>& arr)
// {
//         // request buffer info from numpy array
//     pybind11::buffer_info buf = arr.request();

//     // check if dimensions are equal
//     if (buf.shape[0] != bmat.size1() && buf.shape[1] != bmat.size2())
//     {
//         throw std::runtime_error("Dimensions are not equal!")
//     }
//     rsz = buf.shape[0];
//     csz = buf.shape[1];

//     // set a new pointer ptr to the buffer pointer
//     double *ptr = (double *)buf.ptr;

//     // set pointer to buffer pointer
//     for (int i = 0; i < rsz; ++i)
//     {
//         for (int j = 0; j < csz; ++j)
//         {
//             bmat(i,j) = ptr[i * csz + j];
//         }
//     }
// }

void test_performance(Matrix *myMat_ptr, ublas::matrix<double> bMat)
{
    // test performance
    int max_iter = 10000; // max iterations
    // empty arrays for mean computation
    // std::vector<double> my_normarr(max_iter);
    // std::vector<double> b_normarr(max_iter);
    // std::vector<double> my_timearr(max_iter);
    // std::vector<double> b_normarr(max_iter);
    double my_norm, b_norm, my_avgnorm, b_avgnorm;
    double my_time, b_time, my_avgtime, b_avgtime;

    for (int i = 0; i < max_iter; ++i)
    {
        clock_t my_t, b_t;
        // evaluate norm from matrix class
        my_t = clock();
        double my_n = myMat_ptr->norm();
        // myMat_ptr->norm() = my_n;
        my_t = clock() - my_t;
        // std::cout << "Norm from my library: " << my_n << std::endl
        //          << "Time from my library: " << (double) my_t / CLOCKS_PER_SEC << std::endl;

        // now from ublas
        b_t = clock();
        double b_n = norm_frobenius(bMat);
        b_t = clock() - b_t;
        // std::cout << "Norm from boost: " << b_n << std::endl
        //          << "Time from boost: " << (double) b_t / CLOCKS_PER_SEC << std::endl;

        // append the results
        my_norm += my_n;
        b_norm += b_n;
        my_time += ((double)my_t / CLOCKS_PER_SEC);
        b_time += ((double)b_t / CLOCKS_PER_SEC);
        // my_normarr.push_back(my_norm);
        // b_normarr.push_back(b_norm);
        // my_timearr.push_back((float) my_t / CLOCKS_PER_SEC);
        // b_timearr.push_back((float) b_t / CLOCKS_PER_SEC);
    }

    // evaluate the mean
    my_avgnorm = my_norm / max_iter;
    b_avgnorm = b_norm / max_iter;
    my_avgtime = my_time / max_iter;
    b_avgtime = b_time / max_iter;

    // print results
    printf("Results from performance benchmarking for %d-by-%d matrix:\n", bMat.size1(), bMat.size2());
    printf("Number of iterations for performance benchmark: %d\n", max_iter);
    std::cout << "Average norm evaluated from user-defined matrix: " << my_avgnorm << ", \t Average norm evaluated from Boost: " << b_avgnorm << std::endl
              << "Average time taken for evaluation of norm from user-defined matrix: " << my_avgtime << std::endl
              << "Average time taken for evaluation of norm from Boost: " << b_avgtime << std::endl
              << "Comparisons:" << std::endl
              << "Accuracy of user-defined matrix norm vs Boost norm: " << abs(b_avgnorm - my_avgnorm) << std::endl
              << "Ratio of user-defined matrix performance to Boost performance: " << my_avgtime / b_avgtime << std::endl;
}

int main()
try
{
    // initialize parameters
    // int rs{3};
    // int cs{4};
    // std::string fname{"data/small_data.tsv"};

    int rs{3000};
    int cs{4000};
    std::string fname{"data/large_data.tsv"};

    // create matrices
    Matrix *myMat_ptr = new Matrix{rs, cs, fname};
    ublas::matrix<double> bMat(rs, cs);
    read_from_file(bMat, fname);

    test_performance(myMat_ptr, bMat);

    return 0;
}
catch (std::exception &e)
{
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}

// {
//     // common parameters for both matrices
//     int rs{3};
//     int cs{4};
//     std::string fname{"data/small_data.txt"};

//     // // // common parameters for both matrices
//     // int rs {3000};
//     // int cs {4000};
//     // std::string fname {"data/large_data.txt"};

//     // create matrices and print them out
//     Matrix myMat{rs, cs, fname};
//     // Matrix *myMat = new Matrix {rs, cs, fname};
//     // double my_norm;
//     // myMat->norm() = my_norm;
//     // myMat.print_mat();

//     ublas::matrix<double> bMat(rs, cs);
//     read_from_file(bMat, fname);
//     // std::cout << bMat << '\n';

//     std::cout << "hello" << std::endl;

//     // now compute norm for both and print them out
//     // iterate 1e6 times and take average to determine average time
//     double my_norm = myMat.norm();
//     std::cout << "Norm from my library: " << my_norm << std::endl;

//     double b_norm = norm_frobenius(bMat);
//     std::cout << "Norm from boost: " << b_norm << std::endl;
// }