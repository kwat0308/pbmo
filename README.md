# PBMO - Performance Benchmarks using Matrix Operations

A performance benchmarking tool for different matrix implementations by evaluating matrix operations. This tool displays the benchmarked results as a graph.

Currently this package supports matrix multiplications for the following languages / implementations:

- Python
- NumPy
- C++
- Boost
- CuPy
- pyCUDA
- cuBLAS
- Numba (with and without CPU parallelization)
- Numba with CUDA

## Hard Dependencies

- Python >= 3.0
- NumPy
- tabulate
- PlotLy

## Optional Dependencies

These dependencies **must** be downloaded to observe performance benchmarks for other implementations.

- Boost : Download [Boost](https://www.boost.org/users/download/) and unzip **_in the parent directory of root_**.
- CuPy : Check the [CuPy Documentation](https://docs.cupy.dev/en/stable/install.html) for installation requirements of cupy. Can be installed by using `pip`, but is required to be installed with the appropriate CUDA version.
- pyCUDA : Check the [pyCUDA Wiki](https://wiki.tiker.net/PyCuda/Installation/Windows/) for installation requirements of pyCUDA. Can be installed by `pip install pycuda`.
- cuBLAS : We use the `scikit-cuda` package that utilizes pyCUDA. Check the [scikit-cuda Wiki](https://scikit-cuda.readthedocs.io/en/latest/) for installation requirements. This can be installed with `pip install scikit-cuda`.
- Numba : Can be installed with `pip install numba`. Check [numba documentation](https://numba.readthedocs.io/en/stable/cuda/overview.html) for details.
- Numba with CUDA : Only compatible with NVIDIA graphics card with compute capability 3.0 or above. Check [Numba with CUDA Overview](https://numba.readthedocs.io/en/stable/cuda/overview.html) for more information, as well as necessary configurations to allow Numba to access CUDA Toolkit.

**Note**: Currently requires all such packages to be installed. This will be fixed in a future release.

## Installation

Our lastest release can be downloaded by `pip`:

```
pip install pbmo
```

The most latest (unstable) version can be installed through [GitHub](https://github.com/kwat0308/pbmo):

```
git clone https://github.com/kwat0308/pbmo.git
cd pbmo/
pip install .
```

## Running the Code

The code can be run in any Python interactive shell, as well in a Jupyter notebook. Examples are placed in the `examples/` directory.

A minimal example would be of such:

```
import numpy as np
from pbmo.pbmo import PBMO

ndim = np.arange(10)  # max 10x10 matrix
ndims = list(zip(ndim, ndim))

pbmo = PBMO(dims=ndims)  # 10000 iterations
pbmo.evaluate_matmul()
pbmo.collect_results()
pbmo.plot_results()  # plots log scale
```

**Note**: Running plotly on Jupyter Notebook or Jupyter Lab requires additional configurations. Check [here](https://plotly.com/python/getting-started/) for details.

<!-- This project compares the performance between a built-in array, an external C++ library, and Python (numpy) by evaluating the norm from matrices constructed using C-arrays, the uBLAS library from Boost, and a numpy array.

###### Notes for Window command prompt users
- All python commands (ex. **python setup.py build_ext**) should be replaced with **py** instead.
- A **.exe** extension should be applied instead of a **.o** extension for object codes.

## Requirements
- Python3 or above
- **pip** (if Python version is <=3.4)
- numpy and matplotlib modules in Python
- C++11 or above
- uBLAS library in Boost
- A C++ compiler (g++, CLANG++, or Microsoft C++ compiler)
- pybind11

## Installation
The module containing the relavent classes can be installed in the following ways:
1. using **pip**: **pip install .**
2. using **python**: **python setup.py build_ext**, then **python setup.py install**
The modules for the user-defined Matrix class (matrices created from C-arrays) can be imported with **Matrix**. The module for matrices constructed from Boost can be imported with **BoostMatrix**.
- If a different C++ compiler is being used, make sure to change the compiler configurations (**os.environ["CC"]**) in the top of the file **setup.py**.
- If you do not have **pybind11** installed, install from **pip** by using the following command: **pip install pybind11**. Alternatively, this can be downloaded using **git clone** from [here](https://github.com/pybind/pybind11/tree/stable).
- If you do not have **Boost** installed, install it from [here](https://www.boost.org/doc/libs/1_73_0/more/getting_started/windows.html).

## Testing for performance

Performance tests can be run by **python tests/test_from_nparray.py** and **python tests/test_from_datafile.py**, where tests are run by using numpy arrays or constructed data files as valid inputs respectively. The maximum matrix dimension (n) for a n-by-n matrix is set to 1000. A plot showing the performance as a function of dimension will be displayed. The constructed numpy array / datafile will have random values from 0-1 as elements with the provided shape.
Available flags:
- **--verbosity, -v**: Set verbosity level (integer from 0-4). Default is 0.
- **--debug, -d**: Activate debugging mode. Sets verbosity to level 4 and presets maximum dimension (n) to 50.

Performance test can also be run by **python tests/test_matrix.py**. This compares the performance for a small and large matrix. The rows and columns of the smaller matrix, as well as the scaling factor of the smaller vs larger matrix, is set by user input. The constructed numpy array / datafile will have random values from 0-1 as elements with the provided shape.
Available flags:
- **--verbosity, -v**: Set verbosity level (integer from 0-4). Default is 0.
- **--mode, -m**: Set whether to benchmark performance from a datafile, a numpy array, or both (inputs: datafile, np_array, or both). Default is both.
- **--debug, -d**: Activate debugging mode. Sets verbosity to level 4 and presets rowsize = 3, columnsize = 4, scale = 50.

## Tasks
- [x] Implement performance benchmarks in Python with datafiles
- [x] Implement performance benchmarks in Python with iterations
- [x] Implement constructors that pass numpy arrays by reference
- [x] Integrate Boost performance test with numpy array tests
- [x] Create performance plot (dimension vs time)
- [x] Fix slow performance for C++ matrices
- [ ] Add more descriptions for distribution in setup.py
- [ ] Implement performance benchmarks in C++
- [ ] Create a Makefile for performance benchmarks in C++
- [ ] Add uneven matrices for plotting -->

<!-- Performance tests can be run by **python tests/test_from_nparray.py** and **python tests/test_from_datafile.py**, where tests are run by using numpy arrays or constructed data files as valid inputs respectively. The user input requires the maximum matrix dimension (n) for a square matrix. A plot showing the performance as a function of dimension will be displayed. The constructed numpy array / datafile will have random values from 0-1 as elements with the provided shape.
Available flags:
- **--verbosity, -v**: Set verbosity level (integer from 0-4). Default is 0.
- **--debug, -d**: Activate debugging mode. Sets verbosity to level 4 and presets maximum dimension (n) to 10. -->

<!-- ## Installation of user-defined module
The C++ library containing the user-defined Matrix class can simply be installed by using the usual **pip** command: **pip install .**  .  Alternatively, the package can be built by using the Python command: **python setup.py build_ext**, followed by installing the package with **python setup.py install**.
- If a different C++ compiler is being used, make sure to change the compiler configurations (**os.environ["CC"]**) in the top of the file **setup.py**.
- If you do not have **pybind11** installed, install from **pip** by using the following command: **pip install pybind11**.
The Matrix module can then be used by importing **Matrix**.

## Testing for Python vs C++ performance
To test how the perfomance of a **numpy** array compares with the matrix constructed from the user-defined Matrix class in C++, run **python test/test_matrix.py**. This allows the user to test the performance for the Matrix class vs numpy arrays with row and column dimensions from user input, as well as the scaling factor for larger matrices. The constructed numpy array / datafile will have random values from 0-1 as elements with the provided shape.
Available flags:
- **-v, --verbosity**: Set verbosity level (integer from 0-4). Default is 0.
- **-m, --mode**: Set whether to benchmark performance from a datafile, a numpy array, or both (inputs: datafile, np_array, or both). Default is both.
- **-d, --debug**: Activate debugging mode. Sets verbosity to level 4 and presets rowsize = 3, columnsize = 4, scale = 1000.

## Testing the performance of Boost C++ matrices vs user-defined matrices
To compare between matrices constructed from the **uBLAS** library from **Boost** and the user-defined Matrix class, perform the following steps:
  1. Make sure that the data files **small_data.tsv** and **large_data.tsv** exist in the **data** directory. If not, run **python test/test_matrix.py** after installing the C++ library. Also, make sure that you have installed the **Boost** library from the following website: https://www.boost.org/doc/libs/1_73_0/more/getting_started/windows.html. This will automatically contain the **uBLAS** library.
  2. Compile the source code from the project directory. The command used will differ depending on your compiler.
     - If you are using the Microsoft C++ compiler from command line, run the following command: **cl.exe /EHsc /Fe"tests\test_matrix.exe" /I cpptasks\include /I \path\to\boost tests\test_matrix.cc cpptasks\src\Matrix.cc** (replace **path\to\boost** to your path to the directory in which boost is installed).
     - If you are using a Unix system with terminal (either a g++ or CLANG++ compiler), run the following command: **g++ -o tests/test_matrix.o -I cpptasks/include tests/test_matrix.cc cpptasks/src/Matrix.cc**. For CLANG++, replace **g++** with **clang++**.
  3. Run the executable from the project directory. -->
