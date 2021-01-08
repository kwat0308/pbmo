#!/usr/bin/env python

import sys
import os
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import CCompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import MSVCCompiler

local_path = os.path.dirname(os.path.abspath(__file__))

# Get the path to boost
# Currently i cant find a workaround to this, we can painstakingly take only
# header files we need for uBLAS, but this takes time and effort.
# This also assumes that the Boost directory is in the same directory as the root
# of this module, which needs to be improved (if it can be)


def get_include_boost():
    '''Return path to where Boost is located'''
    root_dir = os.path.dirname(local_path)
    boost_dir = os.path.join(root_dir, "boost_1_75_0/boost_1_75_0")
    return boost_dir


# C/C++ extension of pbmo module
# contains magnetic field and trajactory evaluation as core of the code
libpbmo = Extension(
    'pbmo.lib._libpbmo',
    sources=[
        "pbmo/lib/src/Matrix.cpp", "pbmo/lib/src/BoostMatrix.cpp",
        "pbmo/lib/src/pybind11_wrapper.cpp"
    ],
    language='c++',
    include_dirs=['pbmo/lib/include', 'pbmo/lib/extern',
                  get_include_boost()])
'''
The below settings were obtained from the Iminuit package from scikit-HEP:
https://github.com/scikit-hep/iminuit 
'''
extra_flags = []
if bool(os.environ.get("COVERAGE", False)):
    extra_flags += ["--coverage"]
if platform.system() == "Darwin":
    extra_flags += ["-stdlib=libc++"]

# turn off warnings raised by Minuit and generated Cython code that need
# to be fixed in the original code bases of Minuit and Cython
compiler_opts = {
    CCompiler: {},
    UnixCCompiler: {
        "extra_compile_args": [
            "-std=c++11",
            "-Wno-shorten-64-to-32",
            "-Wno-parentheses",
            "-Wno-unused-variable",
            "-Wno-sign-compare",
            "-Wno-cpp",  # suppresses #warnings from numpy
            "-Wno-deprecated-declarations",
        ] + extra_flags,
        "extra_link_args":
        extra_flags,
    },
    MSVCCompiler: {
        "extra_compile_args": ["/EHsc"]
    },
}


class SmartBuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler
        opts = [v for k, v in compiler_opts.items() if isinstance(c, k)]
        for e in self.extensions:
            for o in opts:
                for attrib, value in o.items():
                    getattr(e, attrib).extend(value)

        build_ext.build_extensions(self)


# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/en/latest/development.html#single-sourcing-the-version-across-setup-py-and-your-project
# This is one of the recommended methods that works in Python 2 and 3:


def get_version():
    version = {}
    with open("pbmo/version.py") as fp:
        exec(fp.read(), version)
    return version['__version__']


__version__ = get_version()

setup(
    name='pbmo',
    version=__version__,
    description='Performance Benchmarking Tool using Matrix Operations',
    packages=['pbmo', 'pbmo.tests', 'pbmo.lib'],
    author='Keito Watanabe',
    author_email='k.wat8973@gmail.com',
    ext_modules=[libpbmo],
    cmdclass={"build_ext": SmartBuildExt},
    install_requires=['numpy', 'matplotlib', 'plotly', 'tabulate'],
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
