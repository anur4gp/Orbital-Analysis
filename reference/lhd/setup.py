from setuptools import setup, Extension
import pybind11

'''
To compile the pybind:


pip install setuptools 

pip install pybind 
    or 
pip install pybind11
    or 
pip install pybind-11

Make sure that your c++ compiler has access to

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> -->> these should already be part of your pybind installation
                               on the newer versions.



#include <random>
#include <vector>
#include <numeric>
#include <cstring>
#include <cmath>
#include <algorithm>   -->> these are part of almost any basic c++ compiler.



py setup.py build_ext --inplace

'''
 
ext = Extension(
    "pt_maxpro",
    sources=["pt_maxpro.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["/std:c++17", "/O2"] if False else None,  
)
 


'''

    Here, you might want to play a little bit with 
    the O3 and O2 tags. If your computer is win64,
    I suggest using O3. It would be a once in a lifetime
    error if something comes up, and the code would be nicely
    optimized.

'''
import sys
if sys.platform == "win32":
    ext.extra_compile_args = ["/std:c++17", "/O2", "/EHsc"]
else:
    ext.extra_compile_args = ["-std=c++17", "-O3"]
 
setup(
    name="pt_maxpro",
    version="0.1.0",
    author="Jaime Beilis",
    description="Parallel tempering MaxPro Latin hypercube design (pybind11 module)",
    ext_modules=[ext],
)
