UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
EIGEN_PATH = /usr/include/eigen3
endif
ifeq ($(UNAME), Darwin)
EIGEN_PATH = /usr/local/include/eigen3
endif

PYTHON_INCLUDE_PATH = /usr/include/python2.7
CXX = g++
CXXFLAGS = -std=c++17 -O3 -I$(EIGEN_PATH) -g

_continuous.so: continuous_wrap.cxx continuous.cpp
	$(CXX) -shared -fPIC -I./ -I$(PYTHON_INCLUDE_PATH) -I$(EIGEN_PATH) -o $@ $^ -lm -lstdc++

continuous_wrap.cxx: continuous.i
	swig -c++ -python -I$(EIGEN_PATH) $<
