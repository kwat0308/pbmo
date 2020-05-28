# cpptasks
Performance benchmark for C++ and Python using a Matrix class

## Requirements
- Python 3 or above
- C++11 or above
- A C++ compiler (g++, CLANG++, or Microsoft C++ compiler)

## Installation of user-defined Matrix module
The C++ library can simply be installed by using the usual **pip** command: **pip install .**.  Alternatively, the package can be install by using a Python command: **python setup.py build_ext**.
- If a different C++ compiler is being used, make sure to comment out the compiler configurations (**os.environ["CC"]**) in the top of the file **setup.py**.

## Testing for Python vs C++ performance
To test how the perfomance of a **numpy** array compares with the matrix constructed from the user-defined Matrix class in C++, run **python test/test_matrix.py**. 

## Testing for Boost C++ matrices vs user-defined matrices performance
To compare between matrices constructed from the **uBLAS** library from **Boost** and the user-defined Matrix class, perform the following steps:
  1. Make sure that the data files **small_data.txt** and **large_data.txt** exist in the **data** directory. If not, run **python test/test_matrix.py** after installing the C++ library.
  2. Compile the source code from the project directory. The command used will differ depending on your compiler. 
    a. If you are using the Microsoft C++ compiler from command line, run the following command: **cl.exe /EHsc /Fe"cpptasks\src\test_matrix.exe" /I cpptasks\include /I ..\..\apps\boost_1_73_0 cpptasks\src\test_matrix.cc cpptasks\src\Matrix.cc**
    b. If you are using a Unix system with terminal (either a g++ or CLANG++ compiler), run the following command: **g++ -o cpptasks/src/test_matrix cpptasks/src/test_matrix.cc cpptasks/src/Matrix.cc**. For CLANG++, replace **g++** with **clang++**.
  3. Run the compiled code by using the following command: **./cpptasks/src/test_matrix.exe** (for Windows C++ compiler) / **./cpptasks/src/test_matrix** (for g++ / CLANG++ compiler).
