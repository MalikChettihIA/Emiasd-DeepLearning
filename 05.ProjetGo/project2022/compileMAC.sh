#!/bin/bash

# make sure pybind11 is installed : `pip3 install pybind11`

#c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` golois.cpp -o golois'python3-config --extension-suffix’

#c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
#-arch arm64 \
#-I/Users/malikchettih/.pyenv/versions/3.9.21/include/python3.9 -I/Users/malikchettih/.pyenv/versions/3.9.21/lib/python3.9/site-packages/pybind11/include \
#golois2.cpp -o golois.cpython-39-darwin.so

export MACOSX_DEPLOYMENT_TARGET=13.0

c++ -O3 -Wall -shared -std=c++11 -fvisibility=hidden -undefined dynamic_lookup \
    $(python3 -m pybind11 --includes) \
    $(python3-config --includes) \
    golois.cpp \
    -o golois$(python3-config --extension-suffix)

