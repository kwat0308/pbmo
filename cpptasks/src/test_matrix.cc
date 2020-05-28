#include <iostream>
#include <fstream>
#include <string>
#include "Matrix.h"

// boost directives
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

void read_from_file(matrix<double> &bmat, const std::string &fname)
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

int main()
{
    // common parameters for both matrices
    int rs{3};
    int cs{4};
    std::string fname{"data/small_data.txt"};

    // // // common parameters for both matrices
    // int rs {3000};
    // int cs {4000};
    // std::string fname {"data/large_data.txt"};

    // create matrices and print them out
    Matrix myMat{rs, cs, fname};
    // Matrix *myMat = new Matrix {rs, cs, fname};
    // myMat.print_mat();

    matrix<double> bMat(rs, cs);
    read_from_file(bMat, fname);
    // std::cout << bMat << '\n';

    std::cout << "hello" << std::endl;

    // now compute norm for both and print them out
    // iterate 1e6 times and take average to determine average time
    double my_norm = myMat.norm();
    std::cout << "Norm from my library: " << my_norm << std::endl;

    double b_norm = norm_frobenius(bMat);
    std::cout << "Norm from boost: " << b_norm << std::endl;
}