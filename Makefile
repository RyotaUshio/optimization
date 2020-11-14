UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
EIGEN_PATH = /usr/include/eigen3
endif
ifeq ($(UNAME), Darwin)
EIGEN_PATH = /usr/local/include/eigen3
endif

CXX = g++
CXXFLAGS = -std=c++17 -O3 -I$(EIGEN_PATH) -g

all: main main2	

%: %.cpp continuous.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm *~ *.o main

.PHONY: all clean
