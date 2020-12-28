#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include "pybind11/stl.h" // for STL container type conversions

#include "Matrix.h" // Matrix header file

namespace py = pybind11;

PYBIND11_MODULE(Matrix, M)
{
  // matrix class
  M.doc() = "Matrix class composed of 2-D arrays.";
  py::class_<Matrix>(M, "Matrix", py::buffer_protocol())
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
}

/*


        // for initialization of matrix using numpy array of type double (buffer protocol required)
        // obtained from https://stackoverflow.com/questions/57990269/passing-pointer-to-c-from-python-using-pybind11
        // .def("__init__", [](Matrix& mat, const int rs, const int cs, py::array_t<double> buffer) {
        // py::buffer_info info = buffer.request();
        // new (&mat) Matrix {info.shape[0], info.shape[1], static_cast<double*>(info.ptr)};
        // })
        
        .def_buffer([](Matrix &m) -> py::buffer_info {    // create buffer protocol for numpy buffers to be passed
        return py::buffer_info(
            m.get_ptr(),                               // Pointer to buffer //
            sizeof(float),                          // Size of one scalar //
            py::format_descriptor<float>::format(), // Python struct-style format descriptor //
            2,                                      // Number of dimensions //
            { m.rows(), m.cols() },                 // Buffer dimensions //
            { sizeof(float) * m.cols(),             // Strides (in bytes) for each index //
              sizeof(float) }
            );
        })
*/