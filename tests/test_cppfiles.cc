
#include "Matrix.h"
#include <iostream>

int main()
{

    std::cout << "Hello\n";
    // double data[12] {1,2,3,4,5,6,7,8,9,0,1,2};
    // Matrix M {3,4, data};
    // M.print();
    Matrix M{3000, 4000, "../data/large_data.txt"};
    // M.print();
    // std::cout << M.rowsize();

    // Matrix M1 {3,4};
    // M1 = M;
    // M1.print();

    // std::cout << M.get_value(1,1);
    double norm = M.norm();
    std::cout << norm;
}