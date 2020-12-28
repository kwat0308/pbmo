/*
Matrix class compiled using nvcc, the NVIDIA CUDA compiler. 
Constructs matrix by memory allocation of each matrix element to each thread.
This is similar to the Matrix class in Matrix.cc
*/

#include "cuMatrix.h"

//default constructor
cuMatrix::cuMatrix(int rowsize, int columnsize)
    : rsz{rowsize}, csz{columnsize}
{
    // allocate in unified memory
    // allocated memory size should be rowsize * columnsize
    // ** not sure if sizeof(double) is necessary
    cudaMallocManaged(&m, rsz*csz*sizeof(double));

    // initialize zero values to each pointer entry
    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            m[i * csz + j] = 0.0; // assign zero value to each index in memory
        }
    }
}

//default constructor
cuMatrix::cuMatrix(int rowsize, int columnsize, const double rand_max)
    : rsz{rowsize}, csz{columnsize}
{
    // allocate in unified memory
    // allocated memory size should be rowsize * columnsize
    // ** not sure if sizeof(double) is necessary
    cudaMallocManaged(&m, rsz*csz*sizeof(double));

    // construct uniform random distribution 
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, rand_max);

    // initialize zero values to each pointer entry
    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            m[i * csz + j] = distribution(generator); // assign zero value to each index in memory
        }
    }
}

// constructor with specified rowsize and column size rs, cs
// where we read values from some dataset
// Assume (for simplicity) that all data are separated by whitespace
// and that there are no headers
// also assume that we know the row and column size of the dataset
cuMatrix::cuMatrix(const int rowsize, const int columnsize, const std::string &fname)
    : rsz{rowsize}, csz{columnsize}
{
    // allocate in unified memory
    // allocated memory size should be rowsize * columnsize
    // ** not sure if sizeof(double) is necessary
    cudaMallocManaged(&m, rsz*csz*sizeof(double));

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
            m[i * csz + j] = val; // set value
            ++j;                  // increment column index
        }
        ++i; // increment row index
    }
}

// // constructor with specified rowsize and columnsize rs, cs
// // where values are obtained from a numpy array
// // we equate the pointers together to pass-by-reference
// // source: https://www.linyuanshi.me/post/pybind11-array/
// cuMatrix::cuMatrix(const pybind11::array_t<double> &arr)
// {
//     // request buffer info from numpy array
//     pybind11::buffer_info buf = arr.request();

//     // set row and column size
//     rsz = buf.shape[0];
//     csz = buf.shape[1];

//     // allocate in unified memory
//     // allocated memory size should be rowsize * columnsize
//     // ** not sure if sizeof(double) is necessary
//     cudaMallocManaged(&m, rsz*csz*sizeof(double))

//     // set a new pointer ptr to the buffer pointer
//     double *ptr = (double *)buf.ptr;

//     // set pointer to buffer pointer
//     for (int i = 0; i < rsz; ++i)
//     {
//         for (int j = 0; j < csz; ++j)
//         {
//             m[i * csz + j] = ptr[i * csz + j];
//         }
//     }
// }

// check if two matrices have the same dimensions
// bool cuMatrix::dim_equal(const cuMatrix &mat)
// {
//     return rsz == mat.nrows() &&
//            csz == mat.ncols();
// }

// // kernel function to evaluate the norm
// __global__
// double cuMatrix::eval_norm(double norm)
// {
//     int index = threadIdx.x;
//     int stride = blockDim.x;
//     for (int i = index; i < rsz*csz; i += stride) {
//         norm += m[i]*m[i];
//     }
// }

// Frobenius norm
double cuMatrix::norm()
{
    double norm = 0.0;

    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            double re = get_value(i,j);
            norm += re * re;
        }
    }

    return sqrt(norm);
}

// // obtain performance of norm by performing max_iter number of
// // evaluations of norm
// // returns pair of average norm and average time
// const std::pair<double, double> cuMatrix::norm_performance(const int max_iter)
// {
//     double avgnorm, avgtime;
//     clock_t t;

//     int i = 0;
//     while (i < max_iter)
//     {
//         // evaluate norm with timer
//         t = clock();
//         double norm_i = norm();
//         t = clock() - t;
//         // append to avgnorm and avgtime
//         avgnorm += norm_i;
//         avgtime += (double)t;
//         ++i;
//     }

//     // divide by ticks / second
//     avgtime /= (CLOCKS_PER_SEC);

//     // get average value
//     avgnorm /= max_iter;
//     avgtime /= max_iter;

//     return std::pair<double, double>(avgnorm, avgtime);
// }

int main()
{
    // Test if we can properly allocate matrix
    // construct matrix with random elements
    cuMatrix test_matrix = cuMatrix(100, 100, 1.0);

    double norm = test_matrix.norm();
    // std::pair<double, double> perf_values = test_matrix.norm_performance(100000);

    // std::cout << perf_values.first << " " << perf_values.second << std::endl;

    // std::cout << test_matrix.nrows() << " " << test_matrix.ncols() << std::endl;
    std::cout << norm << std::endl;
    return 0;
}