// contains class declarations of Matrix class
#ifndef __MATRIX_H_
#define __MATRIX_H_

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <stdexcept>
// #include <time.h>
#include <chrono>
class Matrix
{
private:
    //members
    int rsz;   //row size
    int csz;   //column size
    float *m; // pointer to 2-d array

public:
    //constructor
    Matrix(const int rs, const int cs);
    // Matrix(const int, const int, float data[]);
    Matrix(const int rs, const int cs, const float *mat);
    Matrix(const int rs, const int cs, const std::string &fname);
    //  Matrix(const int, const int, const pybind11::array_t<float>&);
    Matrix(const pybind11::array_t<float> &arr);
    // copy
    Matrix(const Matrix &mat);            // copy constructor
    Matrix &operator=(const Matrix &mat); //copy assignment

    // move
    Matrix(Matrix &&mat);            // move constructor
    Matrix &operator=(Matrix &&mat); // move assignment

    // setters
    void set_value(const int rowindex, const int colindex, const float &newval) { m[rowindex * csz + colindex] = newval; } // change single value

    // getters
    const int rows() const { return rsz; }                                            // number of rows
    const int cols() const { return csz; }                                            // number of columns
    const std::pair<int, int> dim() const { return std::pair<int, int>{rsz, csz}; }   // dimension
    const float get_value(const int i, const int j) const { return m[i * csz + j]; } // value at (i,j) coordinate
    // float* get_ptr() {return m;}   // return pointer of matrix (required for buffer protocol)

    // operations
    bool dim_equal(const Matrix &mat); // check if dimensions are equal
    float inner_prod(const Matrix &mat);
    float norm();                                               //Frobenius norm (standard matrix norm)
    // Matrix matmul(const Matrix &mat);   // matrix multiplication
    std::pair<Matrix, float> matmul(const Matrix &mat, const bool &return_time);   // matrix multiplication
    // const std::pair<float, float> norm_performance(const int max_iter); // evaluate performance through c++
    // utility functions
    void print_mat();          // print matrix
    void print_row(const int rownum); // auxiliary function for print_mat() to print each row
    // Matrix import_data(const int&, const int&, const std::string&);  // import data (knowing data dimensions)
    // destructor
    ~Matrix() { delete[] m; }
};

#endif //__MATRIX_H_