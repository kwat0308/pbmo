
#include "Matrix.hpp"
#include "BoostMatrix.hpp" 
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"         // for STL container type conversions

namespace py = pybind11;

PYBIND11_MODULE(_libpbmo, M) {
  /*
  Extension module for pbmo to C++.
  */
  // M.def("Extension module for pbmo to C++.");
    py::class_<Matrix>(M, "Matrix", py::buffer_protocol(), py::module_local())
        .def(py::init<const int, const int>())
        .def(py::init<const int, const int, const std::string &>())
        // .def(py::init<const int, const int, const py::array_t<double>&>())
        .def(py::init<const py::array_t<double> &>())
        .def_property("value", &Matrix::get_value, &Matrix::set_value)
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def_property_readonly("dim", &Matrix::dim)
        .def("dim_equal", &Matrix::dim_equal)
        .def("inner_prod", &Matrix::inner_prod)
        .def("norm", &Matrix::norm)
        .def("norm_performance", &Matrix::norm_performance)
        .def("print_mat", &Matrix::print_mat);

    py::class_<BoostMatrix>(M, "BoostMatrix", py::buffer_protocol(), py::module_local())
        .def(py::init<const int, const int>())
        .def(py::init<const int, const int, const std::string &>())
        // .def(py::init<const int, const int, const py::array_t<double>&>())
        .def(py::init<const py::array_t<double> &>())
        .def_property("value", &BoostMatrix::get_value, &BoostMatrix::set_value)
        .def_property_readonly("rows", &BoostMatrix::rows)
        .def_property_readonly("cols", &BoostMatrix::cols)
        .def_property_readonly("dim", &BoostMatrix::dim)
        // .def("dim_equal", &BoostMatrix::dim_equal)
        // .def("inner_prod", &BoostMatrix::inner_prod)
        .def("norm_performance", &BoostMatrix::norm_performance)
        .def("norm", &BoostMatrix::norm)
        .def("print_mat", &BoostMatrix::print_mat);

}