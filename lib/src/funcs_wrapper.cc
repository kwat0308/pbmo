#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for STL container type conversions

namespace py = pybind11;

// declarations
int factorial(int);
std::vector<double> squares(int);

// Python binding
PYBIND11_MODULE(funcs, m)
{
    m.doc() = "Contains various functions";
    m.def("factorial", &factorial, "returns factorial of given integer", py::arg("n") = 0);
    m.def("squares", &squares, "returns vector of squares up until given size", py::arg("sz"));
    // return m.ptr();
}
