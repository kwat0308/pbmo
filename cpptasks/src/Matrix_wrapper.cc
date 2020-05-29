#include <pybind11/pybind11.h>
// #include "pybind11/stl.h" // for STL container type conversions

#include "Matrix.h" // Matrix header file

namespace py = pybind11;

PYBIND11_MODULE(Matrix, M)
{
    // matrix class
    M.doc() = "Matrix class composed of 2-D arrays.";
    py::class_<Matrix>(M, "Matrix")
        .def(py::init<const int, const int>())
        .def(py::init<const int, const int, double*>())
        .def(py::init<const int, const int, const std::string &>())
        .def_property("value", &Matrix::get_value, &Matrix::set_value)
        .def_property_readonly("rowsize", &Matrix::rowsize)
        .def_property_readonly("columnsize", &Matrix::columnsize)
        .def_property_readonly("dim", &Matrix::dim)
        .def("dim_equal", &Matrix::dim_equal)
        .def("inner_prod", &Matrix::inner_prod)
        .def("norm", &Matrix::norm)
        .def("print_mat", &Matrix::print_mat);
}