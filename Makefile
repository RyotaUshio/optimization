CXX = g++
CXXFLAGS = -std=c++17 -O3 -I/usr/include/eigen3

all: main

%: %.cpp continuous.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm *~ *.o main

.PHONY: all clean
