#!/usr/bin/env python

import sys
import os

from setuptools import setup, Extension, find_packages

local_path = os.path.dirname(os.path.abspath(__file__))
# change compiler
# os.environ["CC"] = "clang++"
os.environ["CC"] = "g++"

ext_module = [
    Extension(
    'funcs',
    sources=["cpptasks/src/funcs.cc", "cpptasks/src/funcs_wrapper.cc"],
    depends=[],
    language='c++',
    include_dirs=[
        'cpptasks/src'
    ],
    # extra_compile_args=[    # calls error when run by standard C++ compiler (Windows)
    #     '-std=c++14',  
    #     '-Wno-unused-function',
    #     '-Wno-write-strings',
    # ],
),
    Extension(
        'Matrix',
        sources=["cpptasks/src/Matrix.cc", "cpptasks/src/Matrix_wrapper.cc"],
        depends=[],
        language='c++',
        include_dirs=[
            'cpptasks/src'
        ],
        # extra_compile_args=[
        #     '-std=c++14',  
        #     '-Wno-unused-function',
        #     '-Wno-write-strings',
        # ],
    )
]

setup(
    name='myModule',
    version=1.0,
    description='Example module that contains some basic functions and a matrix class.',
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
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Natural Language :: English',
        
    ],
)