#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include "pybind11/stl.h" // for STL container type conversions

#include "BoostMatrix.h" // Matrix header file

namespace py = pybind11;

PYBIND11_MODULE(BoostMatrix, M)
{
    // matrix class
    M.doc() = "Matrix class created using Boost matrices from uBLAS.";
    py::class_<BoostMatrix>(M, "BoostMatrix", py::buffer_protocol())
        .def(py::init<const int, const int>())
        .def(py::init<const int, const int, const std::string &>())
        // .def(py::init<const int, const int, const py::array_t<double>&>())
        .def(py::init<const py::array_t<double>&>())
        .def_property("value", &BoostMatrix::get_value, &BoostMatrix::set_value)
        .def_property_readonly("rows", &BoostMatrix::rows)
        .def_property_readonly("cols", &BoostMatrix::cols)
        .def_property_readonly("dim", &BoostMatrix::dim)
        // .def("dim_equal", &BoostMatrix::dim_equal)
        // .def("inner_prod", &BoostMatrix::inner_prod)
        .def("norm", &BoostMatrix::norm)
        .def("print_mat", &BoostMatrix::print_mat);
}