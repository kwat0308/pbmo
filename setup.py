#!/usr/bin/env python

import sys
import os

from setuptools import setup, Extension, find_packages

local_path = os.path.dirname(os.path.abspath(__file__))
# change compiler
os.environ["CC"] = "clang++"
# os.environ["CC"] = "g++"
# os.environ["CC"] = "cl"


# obtained from: https://github.com/pybind/python_example/blob/master/setup.py
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_module = [
    Extension(
        'funcs',
        sources=["cpptasks/src/funcs.cc", "cpptasks/src/funcs_wrapper.cc"],
        depends=[],
        language='c++',
        include_dirs=[
            get_pybind_include(),
        ]
    ),
    Extension(
        'Matrix',
        sources=["cpptasks/src/Matrix.cc", "cpptasks/src/Matrix_wrapper.cc"],
        depends=[],
        language='c++',
        include_dirs=[
            get_pybind_include(),
            'cpptasks/include'
        ],
    ),
    Extension(
        'BoostMatrix',
        sources=["cpptasks/src/BoostMatrix.cc", "cpptasks/src/BoostMatrix_wrapper.cc"],
        depends=[],
        language='c++',
        include_dirs=[
            get_pybind_include(),
            'cpptasks/include'
        ],
    )
]

setup(
    name='myModule',
    version='1.1.0',
    description=
    'Example module that contains some basic functions and a matrix class.',
    packages=find_packages(exclude='tests'),
    ext_modules=ext_module,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Natural Language :: English',
    ],
)



# extra_compile_args=[    # calls error when run by standard C++ compiler (Windows)
#     '-std=c++14',   # require C++14 or higher
#     '-Wno-unused-function',  # do not raise error even when function is unused
#     '-Wno-write-strings',   
# ],