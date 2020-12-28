/* This code aims to create a class of a matrix (2-d array) with some common properties as member functions.

This code should have a Matrix class that has standard operations using built-in C array as our matrix.
Our custom created Matrix class should: 
    - have the following members:
        - rowsize, columnsize, 2-d array (with template type T) - for now
    - be able to return the dimension of the matrix
    - be able to return the elements of the matrix
        - this means we need to overload operator [] for subscripting
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
    - from external libraries like Boost / Eigen
*/
// #include <numeric>

#include "Matrix.h" // header file

// constructor with specified rowsize and columnsize rs, cs
// default constructor
// initialize a rs x cs matrix (filled with zero value)
Matrix::Matrix(const int rs, const int cs)
    : rsz{rs}, csz{cs}, m{new double[rs * cs]}
{
    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            m[i * csz + j] = 0.0; // assign zero value to each index in memory
        }
    }
}

// constructor with specified rowsize and columnsize rs, cs
// with given data as a 1-d array (2-d array cant be used as parameter unless we know row length)
// initialize a rs x cs matrix
Matrix::Matrix(const int rs, const int cs, const double *dp)
    : rsz{rs}, csz{cs}, m{new double[rs * cs]}
{
    if (sizeof(dp) == sizeof(m)) // if data size == pointer size
    {
        int i = 0;
        int len = rs * cs;
        while (i < len)
        {
            *m++ = *dp++;
            ++i;
        }
    }
    else
    {
        throw std::runtime_error("Pointer dimensions dont match!");
    }
}

// constructor with specified rowsize and columnsize rs, cs
// where values are obtained from a numpy array
// we equate the pointers together to pass-by-reference
// source: https://www.linyuanshi.me/post/pybind11-array/
Matrix::Matrix(const pybind11::array_t<double> &arr)
{
    // request buffer info from numpy array
    pybind11::buffer_info buf = arr.request();

    // set row and column size
    rsz = buf.shape[0];
    csz = buf.shape[1];

    // allocate new space in free store for pointer
    m = new double[rsz * csz];

    // set a new pointer ptr to the buffer pointer
    double *ptr = (double *)buf.ptr;

    // set pointer to buffer pointer
    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            m[i * csz + j] = ptr[i * csz + j];
        }
    }
}

// // constructor with specified rowsize and columnsize rs, cs
// // where values are obtained from a numpy array
// // we equate the pointers together to pass-by-reference
// // source: https://www.linyuanshi.me/post/pybind11-array/
// Matrix::Matrix(const int rs, const int cs, const pybind11::array_t<double>& arr)
//     :rsz{rs}, csz{cs}, m{new double[rs*cs]}
// {
//     // request buffer info from numpy array
//     pybind11::buffer_info buf = arr.request();

//     // now set pointer to pointer of buffer
//     m = (double*) buf.ptr;
// }

// constructor with specified rowsize and column size rs, cs
// where we read values from some dataset
// Assume (for simplicity) that all data are separated by whitespace
// and that there are no headers
// also assume that we know the row and column size of the dataset
Matrix::Matrix(const int rs, const int cs, const std::string &fname)
    : rsz{rs}, csz{cs}, m{new double[rs * cs]}
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
            m[i * csz + j] = val; // set value
            ++j;                  // increment column index
        }

        ++i; // increment row index
    }
}

// // copy constructor
Matrix::Matrix(const Matrix &mat)
    : rsz{mat.rsz}, csz{mat.csz}, m{new double[mat.rsz * mat.csz]}
{
    if (dim_equal(mat))
    {
        std::copy(mat.m, mat.m + (mat.rsz * mat.csz), m);
    }
    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// copy assignment
Matrix &Matrix::operator=(const Matrix &mat)
{
    if (dim_equal(mat))
    {
        double *p = new double[mat.rsz * mat.csz];
        // m = mat.m;
        std::copy(mat.m, mat.m + (mat.rsz * mat.csz), p);
        delete[] m;
        m = p;
        rsz = mat.rsz;
        csz = mat.csz;
        return *this;
    }
    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// move constructor
Matrix::Matrix(Matrix &&mat)
    : rsz{mat.rsz}, csz{mat.csz}, m{new double[mat.rsz * mat.csz]}
{
    if (dim_equal(mat))
    {
        mat.rsz = 0;
        mat.csz = 0;
        mat.m = nullptr;
    }

    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// move assignment
Matrix &Matrix::operator=(Matrix &&mat)
{
    if (dim_equal(mat))
    {
        delete[] m;
        m = mat.m;
        rsz = mat.rsz;
        csz = mat.csz;

        mat.m = nullptr;
        mat.rsz = 0;
        mat.csz = 0;
        return *this;
    }

    else
    {
        throw std::runtime_error("Dimensions must be equal!");
    }
}

// check if two matrices have the same dimensions
bool Matrix::dim_equal(const Matrix &M)
{
    return rsz == M.rows() &&
           csz == M.cols();
}

// inner product between two matrices
double Matrix::inner_prod(const Matrix &M)
{
    double inner_prod{0.};

    if (dim_equal(M))
    {
        for (int i = 0; i < rsz; ++i)
        {
            for (int j = 0; j < csz; ++j)
            {
                inner_prod += m[i * csz + j] * M.get_value(i, j);
            }
        }

        return inner_prod;
    }
    else
    {
        throw std::runtime_error("Dimensions of Matrices not equal!");
    }
}

// norm of matrix
double Matrix::norm()
{
    double norm{0.};

    for (int i = 0; i < rsz; ++i)
    {
        for (int j = 0; j < csz; ++j)
        {
            double re = get_value(i,j);
            norm += re * re;
        }
    }

    return sqrt(norm);
}

// obtain performance of norm by performing max_iter number of
// evaluations of norm
// returns pair of average norm and average time
const std::pair<double, double> Matrix::norm_performance(const int max_iter)
{
    double avgnorm, avgtime;
    clock_t t;

    int i = 0;
    while (i < max_iter)
    {
        // evaluate norm with timer
        t = clock();
        double norm_i = norm();
        t = clock() - t;
        // append to avgnorm and avgtime
        avgnorm += norm_i;
        avgtime += (double)t;
        ++i;
    }

    // divide by ticks / second
    avgtime /= (CLOCKS_PER_SEC);

    // get average value
    avgnorm /= max_iter;
    avgtime /= max_iter;

    return std::pair<double, double>(avgnorm, avgtime);
}

// print matrix
void Matrix::print_mat()
{
    int rszlim{50}; // row/column size limit if "too large"

    std::cout << "Matrix (" << rsz << "-by-" << csz << "): " << std::endl;
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

void Matrix::print_row(const int rownum)
{
    int cszlim{50}; // column size limit

    std::cout << "\t{";
    if (csz > cszlim)
    { // for large columns print using ... notation
        for (int j = 0; j < 3; ++j)
        {
            std::cout << get_value(rownum, j) << ' ';
        }
        std::cout << "... ";
        for (int j = 3; j > 0; --j)
        {
            std::cout << get_value(rownum, csz - j) << ' ';
        }
    }
    else
    { // otherwise print the whole matrix
        for (int j = 0; j < csz; ++j)
        {
            std::cout << get_value(rownum, j) << ' ';
        }
    }

    std::cout << '}' << std::endl;
}

// // constructor with specified rowsize and columnsize rs, cs
// // where values are obtained from any pointer
// Matrix::Matrix(const int rs, const int cs, double* arr)
//     :rsz{rs}, csz{cs}, m{new double[rs*cs]}
// {
//     for (int i = 0; i < rsz; ++i)
//     {
//         for (int j = 0; j < csz; ++j)
//         {
//             m[i * csz + j] = arr[i * csz + j]; // assign zero value to each index in memory
//         }
//     }
// }

// constructor with specified rowsize and column size rs, cs
// where we read values from some dataset
// Let's assume (for simplicity) that all data are separated by comma
// and that there are no headers
// also that they are .txt files and no other format
// also assume that we know the row and column size of the dataset
// Matrix Matrix::import_data(const int rs, const int cs, const std::string& fname)
// {
//     std::ifstream ifs {fname};
//     Matrix M {rs, cs};

//     double val;
//     char comma;

//     std::string line, ss;
//     int i = 0;   // row index

//     while (std::getline(ifs, line)) {   // read each line in .csv file
//         int j=0; // column index
//         std::stringstream ss {line};  // stringstream of each row

//         while (ss >> val >> comma) {
//             M.set_value(i,j,val);
//             if (comma!=',') {throw std::runtime_error("Not comma delimited!");}

//             ++j;  // increment column index
//         }

//         ++i;   // increment row index
//     }

//     // for (int i=0; i<rs; ++i) {
//     //     for (int j=0; j<cs; ++i) {
//     //         // ifs >> val >> comma;  // read line
//     //         // if (!ifs || comma!=',') {  // character has to be comma
//     //         //     throw std::runtime_error("Cannot read file!");
//     //         // }
//     //         ifs >> val;  // read one element
//     //         std::cout << val;
//     //         if (!ifs) {
//     //             throw std::runtime_error("Cannot read file!");
//     //         }
//     //         m[i*csz + j] = val;   // assign pointer value to read value
//     //     }
//     // }

// }

/*
// input operator for importing csv files
std::istream& operator>>(std::istream& is, std::vector<double> vec)
{
    char comma;
    double data;
    while (comma!='\n') {
        is >> data >> comma;
        if (!is) return is;
        if (comma != ',') {
            is.clear(std::ios_base::failbit);
            return is;
        }
        vec.push_back(data);
    }

    return is;
}

// // import data from a .txt file named fname and return an initialized Matrix object
// // Let's assume (for simplicity) that all data are separated by commas (,)
// // and that there are no headers 
// // also that they are .txt files and no other format
// // also assume that we know the row and column size of the dataset
// Matrix Matrix::import_data(const int& rs, const int& cs, const std::string& fname)
// {
//     std::ifstream ifs {fname};
//     Matrix M {rs, cs};

//     double val;
//     char comma;

//     for (int i=0; i<rs; ++i) {
//         for (int j=0; j<cs; ++i) {
//             ifs >> val >> comma;  // read line
//             if (!ifs || comma!=',') {  // character has to be comma
//                 throw std::runtime_error("Cannot read file!");
//             }
//             M.set_value(i, j, val);  // set value at (i,j) to read data
//         }
//     }
// }


// // check if two matrices have the same dimensions
// bool dim_equal(const Matrix& M1, const Matrix& M2) 
// {
//     return M1.rows() == M2.rows() && \
//             M1.cols() == M2.cols();
// }

// // inner product between two matrices
// double inner_prod(const Matrix& M1, const Matrix& M2)
// {
//     std::vector<double> ipvec;

//     if (dim_equal(M1, M2)) {
//         for (int i=0; i<M1.rows(); ++i) {
//             for (int j=0; j<M1.cols(); ++j) {
//                 ipvec.push_back(M1.value(i,j) * M2.value(i,j));
//             }
//         }

//         double inner_prod = std::accumulate(ipvec.begin(), ipvec.end(), 0.); //sum elements
//         return inner_prod;
//     }
//     else {
//         throw std::runtime_error('Dimensions of matrices not equal!');
//     }
// }
*/
