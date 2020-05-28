# cpptasks
Performance benchmark for C++ and Python using a Matrix class

###### Notes for Window command prompt users
- All python commands (ex. **python setup.py build_ext**) should be replaced with **py** instead. 
- A **.exe** extension should be applied instead of a **.o** extension for object codes.

## Requirements
- Python3 or above
- numpy and matplotlib modules in Python
- C++11 or above
- uBLAS library in Boost
- A C++ compiler (g++, CLANG++, or Microsoft C++ compiler)
- pybind11 

## Installation of user-defined Matrix module
The C++ library can simply be installed by using the usual **pip** command: **pip install .**  .  Alternatively, the package can be install by using a Python command: **python setup.py build_ext**.
- If a different C++ compiler is being used, make sure to change the compiler configurations (**os.environ["CC"]**) in the top of the file **setup.py**.
- If you do not have pybind11 installed, install by using the following command: **pip install pybind11**.

## Testing for Python vs C++ performance
To test how the perfomance of a **numpy** array compares with the matrix constructed from the user-defined Matrix class in C++, run **python test/test_matrix.py**. 

## Testing for Boost C++ matrices vs user-defined matrices performance
To compare between matrices constructed from the **uBLAS** library from **Boost** and the user-defined Matrix class, perform the following steps:
  1. Make sure that the data files **small_data.txt** and **large_data.txt** exist in the **data** directory. If not, run **python test/test_matrix.py** after installing the C++ library. Also, make sure that you have installed the **Boost** library from the following website: https://www.boost.org/doc/libs/1_73_0/more/getting_started/windows.html. This will automatically contain the **uBLAS** library.
  2. Compile the source code from the project directory. The command used will differ depending on your compiler. 
     - If you are using the Microsoft C++ compiler from command line, run the following command: **cl.exe /EHsc /Fe"cpptasks\src\test_matrix.exe" /I cpptasks\include /I \path\to\boost cpptasks\src\test_matrix.cc cpptasks\src\Matrix.cc** (replace **path\to\boost** to your path to the directory in which boost is installed).
     - If you are using a Unix system with terminal (either a g++ or CLANG++ compiler), run the following command: **g++ -o cpptasks/src/test_matrix.o cpptasks/src/test_matrix.cc cpptasks/src/Matrix.cc**. For CLANG++, replace **g++** with **clang++**.
  3. Run the object code from the project directory.
