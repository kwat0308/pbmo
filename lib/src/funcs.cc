// contains various functions

#include <vector>
#include <stdexcept>
// returns factorial of given integer n
int factorial(int n)
{
    // error if negative
    if (n < 0)
    {
        throw std::runtime_error("Integer must be >= 0!");
    }

    if (n <= 1)
    {
        return 1;
    }
    else
    {
        return n * factorial(n - 1);
    }
}

// returns vector of squares up until given size sz
std::vector<double> squares(int sz)
{
    std::vector<double> vec(sz);
    for (int i = 0; i < sz; ++i)
    {
        vec[i] = i * i;
    }
    return vec;
}

// namespace py = pybind11;

// // Python binding
// // PYBIND11_MODULE(funcs, m) {
// PYBIND11_PLUGIN(funcs) {
//     py::module m("funcs", "Contains various functions");
//     // m.doc() = "Contains various functions";
//     m.def("factorial", &factorial, "returns factorial of given integer", py::arg("n")=0);
//     m.def("squares", &squares, "returns vector of squares up until given size", py::arg("sz"));
//     return m.ptr();
// }
