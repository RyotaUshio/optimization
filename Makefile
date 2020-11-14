CXX = g++
CXXFLAGS = -std=c++17 -O3 -I/usr/include/eigen3 -g
# If you are using MacOS, this should be
# CXXFLAGS = -std=c++17 -O3 -I/usr/local/include/eigen3 -g

all: main main2	

%: %.cpp continuous.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm *~ *.o main

.PHONY: all clean
