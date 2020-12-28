/*
Matrix class compiled using nvcc, the NVIDIA CUDA compiler. 
Constructs matrix by memory allocation of each matrix element to each thread.
This is similar to the Matrix class in Matrix.cc
*/

#ifndef __CUMATRIX_H__
#define __CUMATRIX_H__

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
#include <math.h>
#include <stdexcept>
#include <time.h>
#include <random>

class cuMatrix
{
    private:
        int rsz;   // number of rows
        int csz;   // number of columns
        double *m; // pointer to 2-d array

        /*
        Kernel function for norm evaluation
        */
        // double eval_norm(double norm);
    
    public:
        /*
        Construct a N x M matrix by allocating memory to each thread of the GPU. 
        Default Constructor: Assign zero value to each entry of the matrix.
        Note: the rowsize and columnsize must be limited by the max. number of
                threads in the particular GPU, any value above that should raise
                an error.
        */
        cuMatrix(int rowsize, int columnsize);
        /*
        Construct a N x M matrix by allocating memory to each thread of the GPU. 
        Values are assigned randomly from a uniform distribution in [0,rand_max). 
        This is used so that we can test norm performance without Python
        */
        cuMatrix(int rowsize, int columnsize, const double rand_max);
        /*
        /*
        Construct a N x M matrix by allocating memory to each thread of the GPU. 
        Values are assigned from a data file with given data. 
        */
        cuMatrix(int rowsize, int columnsize, const std::string &fname);
        /*
        Construct a N x M matrix by allocating memory to each thread of the GPU. 
        Values are assigned from an numpy array, with the dimensions automatically
        detected from arr.shape
        */
        // cuMatrix(const pybind11::array_t<double> &arr);
        /*
        Copy matrix elements to this matrix from another matrix
        This can be done both by assignment or by initialization
        */
        // cuMatrix(const cuMatrix &mat);            // copy constructor
        // cuMatrix &operator=(const cuMatrix &mat); //copy assignment
        // /*
        // Move matrix elements to this matrix from another matrix
        // This can be done both by assignment or by initialization
        // - Difference from copy is that it only configures the elements
        //   rather than destructing and constructing a new object
        // */
        // cuMatrix(cuMatrix &&mat);            // move constructor
        // cuMatrix &operator=(cuMatrix &&mat); // move assignment

        /* Number of Rows*/
        const int nrows() { return rsz; }  
        /* Number of Columns*/                                          
        const int ncols() { return csz; }      
        /* Matrix Dimension (rowsize, columnsize)*/                                      
        const std::pair<int, int> dim() { return std::pair<int, int>{rsz, csz}; }   
        /* The matrix element at the (i,j)th entry*/
        const double get_value(const int i, const int j) { return m[i * csz + j]; } 

        // // operations
        // bool dim_equal(const cuMatrix &mat); // check if dimensions are equal
        /*
        The Matrix product between two matrices A(NxM) and B(MxQ). Returns a new matrix equal to dimension NxQ
        ***Will be constructed in future implementations***
        */
        // cuMatrix mat_mul(const cuMatrix &mat);
        /*
        The Frobenius norm (standard matrix norm) of this matrix.
        This function runs the kernel eval_norm and returns the result. 
        */
        double norm();
        const std::pair<double, double> norm_performance(const int max_iter); // evaluate performance through c++

        // utility functions
        // void print_mat();          // print matrix
        // void print_row(const int rownum); // auxiliary function for print_mat() to print each row       
        /*
        Destructor: Free allocated memory in GPU
        */
        ~cuMatrix() { cudaFree(m); } 

};


#endif //__CUMATRIX_H__