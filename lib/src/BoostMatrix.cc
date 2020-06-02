/* This code contains a class that constructs a Boost matrix and fills it with appropriate data,
either from a numpy array or from some datafile.
*/

#include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "BoostMatrix.h"

using namespace boost::numeric;

// constructor with specified rowsize and columnsize rs, cs
// default constructor
// initialize a rs x cs matrix (filled with zero value)
BoostMatrix::BoostMatrix(const unsigned long rs, const unsigned long cs)
    : rsz{rs}, csz{cs}, b{ublas::matrix<double>{rsz, csz}}
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
BoostMatrix::BoostMatrix(const pybind11::array_t<double> &arr)
{
    // request buffer info from numpy array
    pybind11::buffer_info buf = arr.request();

    // set row and column size
    rsz = buf.shape[0];
    csz = buf.shape[1];

    // set matrix
    b = ublas::matrix<double>{rsz, csz};

    // set a new pointer ptr to the buffer pointer
    double *ptr = (double *)buf.ptr;

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
    : rsz{rs}, csz{cs}, b{ublas::matrix<double>{rsz, csz}}
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
    if (dim_equal(mat))
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
    if (dim_equal(mat))
    {
        b = bmat.b;
        rsz = bmat.rsz;
        csz = bmat.csz;
        bmat.b.clear();
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
// double BoostMatrix::inner_prod(const BoostMatrix &M)
// {
//     double inner_prod{0.};

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
// double BoostMatrix::norm()
// {
//     double norm{0.};

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
