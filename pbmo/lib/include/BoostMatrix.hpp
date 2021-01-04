#ifndef __BOOSTMATRIX_H__
#define __BOOSTMATRIX_H__

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace boost::numeric;

class BoostMatrix
{
private:
    unsigned long rsz;
    unsigned long csz;
    ublas::matrix<float> b;

public:
    BoostMatrix(const unsigned long rs, const unsigned long cs);
    BoostMatrix(const pybind11::array_t<float> &arr);
    BoostMatrix(const unsigned long rs, const unsigned long cs, const std::string &fname);

    // copy
    BoostMatrix(const BoostMatrix &mat);            // copy constructor
    BoostMatrix &operator=(const BoostMatrix &mat); //copy assignment

    // move
    BoostMatrix(BoostMatrix &&mat);            // move constructor
    BoostMatrix &operator=(BoostMatrix &&mat); // move assignment

    // // setters
    void set_value(const int i, const int j, const float &newval) { b(i, j) = newval; } // change single value

    // getters
    const unsigned long rows() const { return rsz; }                                                                        // number of rows
    const unsigned long cols() const { return csz; }                                                                        // number of columns
    const std::pair<unsigned long, unsigned long> dim() const { return std::pair<unsigned long, unsigned long>{rsz, csz}; } // dimension
    const float get_value(const int i, const int j) const { return b(i, j); }                                               // value at (i,j) coordinate

    // operations
    bool dim_equal(const BoostMatrix &mat); // check if dimensions are equal
    // float inner_prod(const BoostMatrix & bmat) { return ublas::inner_prod(b, bmat.b); }
    float norm() { return norm_frobenius(b); }                          //Frobenius norm
    BoostMatrix matmul(const BoostMatrix &mat);                         //matrix multiplication
    const std::pair<float, float> norm_performance(const int max_iter); // evaluate performance through c++
    // utility functions
    void print_mat();                 // print BoostMatrix
    void print_row(const int rownum); // auxiliary function for print_mat() to print each row
    // BoostMatrix import_data(const int&, const int&, const std::string&);  // import data (knowing data dimensions)
    // destructor
    ~BoostMatrix() { b.clear(); }
};

#endif //__BOOSTMATRIX_H__