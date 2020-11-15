PYTHON_VERSION = 3.6

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
EIGEN_PATH = /usr/include/eigen3
PYTHON_INCLUDE_PATH = /usr/include/python$(PYTHON_VERSION)
endif
ifeq ($(UNAME), Darwin)
EIGEN_PATH = /usr/local/include/eigen3
PYTHON_INCLUDE_PATH = /Users/ryotaushio/.pyenv/versions/anaconda3-2019.07/include/python3.7m
endif

CXX = g++


_continuous.so: continuous_wrap.cxx ../tmp/continuous.cpp
	$(CXX) -shared -fPIC -I./ -I$(PYTHON_INCLUDE_PATH) -I$(EIGEN_PATH) -o $@ $^ -lm -lstdc++

continuous_wrap.cxx: continuous.i
	swig -c++ -python -I$(EIGEN_PATH) $<

clean:
	rm *.so *.cxx *~ *.py
