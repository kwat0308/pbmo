// contains class declarations of Matrix class
#ifndef __MATRIX_H_
#define __MATRIX_H_

class Matrix
{
private:
    //members
    int rsz;   //row size
    int csz;   //column size
    double *m; // pointer to 2-d array

public:
    //constructor
    Matrix(const int rs, const int cs);
    // Matrix(const int, const int, double data[]);
    Matrix(const int rs, const int cs, const double *mat);
    Matrix(const int rs, const int cs, const std::string &fname);
    //  Matrix(const int, const int, const pybind11::array_t<double>&);
    Matrix(const pybind11::array_t<double> &arr);
    // copy
    Matrix(const Matrix &mat);            // copy constructor
    Matrix &operator=(const Matrix &mat); //copy assignment

    // move
    Matrix(Matrix &&mat);            // move constructor
    Matrix &operator=(Matrix &&mat); // move assignment

    // setters
    void set_value(const int rowindex, const int colindex, const double &newval) { m[rowindex * csz + colindex] = newval; } // change single value

    // getters
    const int rows() const { return rsz; }                                            // number of rows
    const int cols() const { return csz; }                                            // number of columns
    const std::pair<int, int> dim() const { return std::pair<int, int>{rsz, csz}; }   // dimension
    const double get_value(const int i, const int j) const { return m[i * csz + j]; } // value at (i,j) coordinate
    // double* get_ptr() {return m;}   // return pointer of matrix (required for buffer protocol)

    // operations
    bool dim_equal(const Matrix &mat); // check if dimensions are equal
    double inner_prod(const Matrix &mat);
    double norm();                                               //Frobenius norm (standard matrix norm)
    const std::pair<double, double> norm_performance(const int max_iter); // evaluate performance through c++
    // utility functions
    void print_mat();          // print matrix
    void print_row(const int rownum); // auxiliary function for print_mat() to print each row
    // Matrix import_data(const int&, const int&, const std::string&);  // import data (knowing data dimensions)
    // destructor
    ~Matrix() { delete[] m; }
};

#endif //__MATRIX_H_