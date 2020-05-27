# cpptasks
Performance benchmark for C++ and Python using a Matrix class 

## Installation
The C++ library can simply be installed by using usual **pip** command: **pip install .** 

## Testing for Python vs C++ performance
To test how the perfomance of a **numpy** array compares with the matrix constructed from the user-defined Matrix class in C++, we need to run **python test/test_matrix.py**. 

## Testing for Boost C++ matrices vs user-defined matrices performance
To compare between matrices constructed from the **uBLAC** library from **Boost** and the user-defined Matrix class, perform the following steps:
  1. Make sure that the data files **small_data.txt** and **large_data.txt** exist in the **data** directory. If not, run **python test/test_matrix.py** after installing the C++ library.
  2. Run the following command: **g++ -o cpptasks/src/test_matrix cpptasks/src/test_matrix.cc cpptasks/src/Matrix.cc**
  3. Run the compiled code by using the following command: **./cpptasks/src/test_matrix**.
