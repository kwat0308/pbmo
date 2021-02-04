/* This code contains a class that constructs a Boost matrix and fills it with appropriate data,
either from a numpy array or from some datafile.
*/

#include "BoostMatrix.hpp"

using namespace boost::numeric;

// constructor with specified rowsize and columnsize rs, cs
// default constructor
// initialize a rs x cs matrix (filled with zero value)
BoostMatrix::BoostMatrix(const unsigned long rs, const unsigned long cs)
    : rsz{rs}, csz{cs}, b{ublas::matrix<float>{rsz, csz}}
{
    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            b(i, j) = 0.0; // assign zero value to each value
        }
    }
}

// constructor with specified rowsize and columnsize rs, cs
// where values are obtained from a numpy array
// we equate the pointers together to pass-by-reference
// source: https://www.linyuanshi.me/post/pybind11-array/
BoostMatrix::BoostMatrix(const pybind11::array_t<float> &arr)
{
    // request buffer info from numpy array
    pybind11::buffer_info buf = arr.request();

    // set row and column size
    rsz = buf.shape[0];
    csz = buf.shape[1];

    // set matrix
    b = ublas::matrix<float>{rsz, csz};

    // set a new pointer ptr to the buffer pointer
    float *ptr = (float *)buf.ptr;

    // set pointer to buffer pointer
    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            b(i, j) = ptr[i * csz + j];
        }
    }
}

// constructor with specified rowsize and column size rs, cs
// where we read values from some dataset
// Assume (for simplicity) that all data are separated by whitespace
// and that there are no headers
// also assume that we know the row and column size of the dataset
BoostMatrix::BoostMatrix(const unsigned long rs, const unsigned long cs, const std::string &fname)
    : rsz{rs}, csz{cs}, b{ublas::matrix<float>{rsz, csz}}
{
    std::ifstream ifs{fname};
    if (!ifs)
    {
        throw std::runtime_error("Cannot open file!");
    }

    float val;
    std::string line;
    int i = 0; // row index

    while (std::getline(ifs, line))
    {                                // read each line in file
        int j = 0;                   // column index
        std::istringstream ss{line}; // stringstream of each row
        while (ss >> val)
        {
            b(i, j) = val; // set value
            ++j;           // increment column index
        }

        ++i; // increment row index
    }
}

// copy constructor
BoostMatrix::BoostMatrix(const BoostMatrix &bmat)
    : rsz{bmat.rsz}, csz{bmat.csz}
{
    if (dim_equal(bmat))
    {
        b = bmat.b;
    }
    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// copy assignment
BoostMatrix &BoostMatrix::operator=(const BoostMatrix &bmat)
{
    if (dim_equal(bmat))
    {
        b = bmat.b;
        rsz = bmat.rsz;
        csz = bmat.csz;
        return *this;
    }
    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// move constructor
BoostMatrix::BoostMatrix(BoostMatrix &&bmat)
    : rsz{bmat.rsz}, csz{bmat.csz}, b{bmat.b}
{
    if (dim_equal(bmat))
    {
        bmat.rsz = 0;
        bmat.csz = 0;
        bmat.b.clear();
    }

    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// move assignment
BoostMatrix &BoostMatrix::operator=(BoostMatrix &&bmat)
{
    if (dim_equal(bmat))
    {
        b = bmat.b;
        rsz = bmat.rsz;
        csz = bmat.csz;

        bmat.b.clear();
        bmat.rsz = 0;
        bmat.csz = 0;
        return *this;
    }

    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// check if two matrices have the same dimensions
bool BoostMatrix::dim_equal(const BoostMatrix &M)
{
    return rsz == M.rows() &&
           csz == M.cols();
}

// matrix multiplication
// BoostMatrix BoostMatrix::matmul(const BoostMatrix &mat)
// {
//     if (csz != mat.csz)
//     {
//         throw std::runtime_error("Column Dimension of self does not match row dimension of matrix.");
//     }

//     else
//     {
//         BoostMatrix result = BoostMatrix(rsz, mat.csz);

//         result.b = prod(b, mat.b);
//         // return result and the time if return_time is true
//         return result;
//     }
// }

// matrix multiplication
std::pair<BoostMatrix, float> BoostMatrix::matmul(const BoostMatrix &mat, const bool &return_time)
{
    if (csz != mat.csz)
    {
        throw std::runtime_error("Column Dimension of self does not match row dimension of matrix.");
    }

    else
    {
        // evaluate performance of each operation
        auto start = std::chrono::high_resolution_clock::now();
        BoostMatrix result = BoostMatrix(rsz, mat.csz);
        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> elapsed = stop - start;

        float time = elapsed.count();

        result.b = prod(b, mat.b);
        // return result and the time if return_time is true
        return std::pair<BoostMatrix, float>(result, time);
    }
}

// // obtain performance of norm by performing max_iter number of
// // evaluations of norm
// // returns pair of average norm and average time
// const std::pair<float, float> BoostMatrix::norm_performance(const int max_iter)
// {
//     float avgnorm, avgtime;
//     clock_t t;

//     int i = 0;
//     while (i < max_iter)
//     {
//         // evaluate norm with timer
//         t = clock();
//         float norm_i = norm();
//         t = clock() - t;
//         // append to avgnorm and avgtime
//         avgnorm += norm_i;
//         avgtime += (float)t;
//         ++i;
//     }

//     // divide by ticks / second
//     avgtime /= (CLOCKS_PER_SEC);

//     // get average value
//     avgnorm /= max_iter;
//     avgtime /= max_iter;

//     return std::pair<float, float>(avgnorm, avgtime);
// }

// print BoostMatrix
void BoostMatrix::print_mat()
{
    int rszlim{50}; // row/column size limit if "too large"

    std::cout << "BoostMatrix (" << rsz << "-by-" << csz << "): " << std::endl;
    std::cout << '{' << std::endl;
    if (rsz > rszlim)
    { // for large rows print using ... notation
        for (int i = 0; i < 3; ++i)
        {
            print_row(i);
        }
        std::cout << "\t..." << std::endl;
        for (int j = 3; j > 0; --j)
        {
            print_row(rsz - j);
        }
    }
    else
    { // otherwise print the whole matrix
        for (int i = 0; i < rsz; ++i)
        {
            print_row(i);
        }
    }
    std::cout << '}' << std::endl;
}

void BoostMatrix::print_row(const int i)
{
    int cszlim{50}; // column size limit

    std::cout << "\t{";
    if (csz > cszlim)
    { // for large columns print using ... notation
        for (int j = 0; j < 3; ++j)
        {
            std::cout << get_value(i, j) << ' ';
        }
        std::cout << "... ";
        for (int j = 3; j > 0; --j)
        {
            std::cout << get_value(i, csz - j) << ' ';
        }
    }
    else
    { // otherwise print the whole matrix
        for (int j = 0; j < csz; ++j)
        {
            std::cout << get_value(i, j) << ' ';
        }
    }

    std::cout << '}' << std::endl;
}

/*

// // inner product between two matrices
// float BoostMatrix::inner_prod(const BoostMatrix &M)
// {
//     float inner_prod{0.};

//     if (dim_equal(M))
//     {
//         for (int i = 0; i < rsz; ++i)
//         {
//             for (int j = 0; j < csz; ++j)
//             {
//                 inner_prod += m[i * csz + j] * M.get_value(i, j);
//             }
//         }

//         return inner_prod;
//     }
//     else
//     {
//         throw std::runtime_error("Dimensions of Matrices not equal!");
//     }
// }

// // norm of BoostMatrix
// float BoostMatrix::norm()
// {
//     float norm{0.};

//     for (int i = 0; i < rsz; ++i)
//     {
//         for (int j = 0; j < csz; ++j)
//         {
//             norm += abs(get_value(i, j) * get_value(i, j));
//         }
//     }

//     return sqrt(norm);
// }



*/
