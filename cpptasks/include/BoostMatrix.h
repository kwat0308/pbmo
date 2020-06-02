#ifndef __BOOSTMATRIX_H__
#define __BOOSTMATRIX_H__

#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric;

class BoostMatrix
{
private:
    unsigned long rsz;
    unsigned long csz;
    ublas::matrix<double> b;

public:
    BoostMatrix(const unsigned long, const unsigned long);
    BoostMatrix(const pybind11::array_t<double> &);
    BoostMatrix(const unsigned long, const unsigned long, const std::string &);

    // copy 
    BoostMatrix(const BoostMatrix &);            // copy constructor
    BoostMatrix &operator=(const BoostMatrix &); //copy assignment

    // move
    BoostMatrix(BoostMatrix &&);            // move constructor
    BoostMatrix &operator=(BoostMatrix &&); // move assignment

    // // setters
    void set_value(const int i, const int j, const double &newval) { b(i, j) = newval; } // change single value

    // getters
    const unsigned long rows() const { return rsz; }                                                                        // number of rows
    const unsigned long cols() const { return csz; }                                                                        // number of columns
    const std::pair<unsigned long, unsigned long> dim() const { return std::pair<unsigned long, unsigned long>{rsz, csz}; } // dimension
    const double get_value(const int i, const int j) const { return b(i, j); }                                              // value at (i,j) coordinate

    // operations
    bool dim_equal(const BoostMatrix &); // check if dimensions are equal
    double inner_prod(const BoostMatrix & bmat) { return inner_prod(b, bmat.b)};
    double norm() { return norm_frobenius(b); }; //Frobenius norm
    // utility functions
    void print_mat();          // print BoostMatrix
    void print_row(const int); // auxiliary function for print_mat() to print each row
    // BoostMatrix import_data(const int&, const int&, const std::string&);  // import data (knowing data dimensions)
    // destructor
    ~BoostMatrix() {b.clear()}
};

#endif //__BOOSTMATRIX_H__