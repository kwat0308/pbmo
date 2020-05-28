// contains class declarations of Matrix class
#ifndef __MATRIX_H_
#define __MATRIX_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdexcept>
#include <ios>
#include <numeric>
// #include <Eigen/Dense>

/* This code should have a Matrix class that has standard operations using vector<vector<T> > as our matrix.
Our custom created Matrix class should: 
    - have the following members:
        - rowsize, columnsize, 2-d array (with template type T) - for now
    - be able to return the dimension of the matrix
    - be able to return the elements of the matrix
        - this means we need to overload operator [] for subscripting (maybe not, since we are using std::vectors)
    - be able to assign elements to a variable (i.e. int var = mat[0][0])
    - be able to copy elements from some literal / variable (i.e. mat[1][2] = 10)
    - be able to assign a Matrix to another Matrix
    - be able to perform standard operations
        - addition and subtraction (need row and column size to be equal)
        - multiplication
        - elementwise multiplication and division
        - inner product
        - norm
        - power
    - be able to initialize using some input file
    - have a destructor and constructor

With this, we can compare our Matrix class to:
    - numpy array constructed matrices
    - matrices constructed using a standard naive C++ loop
    - from external libraries like Boost / Eigen

*/

class Matrix
{
private:
    //members
    const int rsz;   //row size
    const int csz;   //column size
    const double *m; // pointer to 2-d array

public:
    //constructor
    Matrix(const int, const int);
    Matrix(const int, const int, double data[]);
    Matrix(const int, const int, const std::string &);
    // copy
    Matrix(const Matrix &);            // copy constructor
    Matrix &operator=(const Matrix &); //copy assignment

    // move
    Matrix(Matrix &&);            // move constructor
    Matrix &operator=(Matrix &&); // move assignment

    // setters
    void set_value(const int i, const int j, const double &newval) { m[i * csz + j] = newval; } // change single value

    // getters
    const int rowsize() const { return rsz; }                                         // number of rows
    const int columnsize() const { return csz; }                                      // number of columns
    const std::pair<int, int> dim() const { return std::pair<int, int>{rsz, csz}; }   // dimension
    const double get_value(const int i, const int j) const { return m[i * csz + j]; } // value at (i,j) coordinate

    // operations
    bool dim_equal(const Matrix &); // check if dimensions are equal
    double inner_prod(const Matrix &);
    double norm(); //Frobenius norm (standard matrix norm)
    // utility functions
    void print_mat();            // print matrix
    void print_row(const int &); // auxiliary function for print_mat() to print each row
    // Matrix import_data(const int&, const int&, const std::string&);  // import data (knowing data dimensions)
    // destructor
    ~Matrix() { delete[] m; }
};

#endif