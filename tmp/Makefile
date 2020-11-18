UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
EIGEN_PATH = /usr/include/eigen3
endif
ifeq ($(UNAME), Darwin)
EIGEN_PATH = /usr/local/include/eigen3
endif

CXX = g++
CXXFLAGS = -std=c++17 -O3 -I$(EIGEN_PATH) -g

all: main main2 run

run: main main2
	./main
	./main2

main: main.o continuous.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main2: main2.o continuous.o
	$(CXX) $(CXXFLAGS) -o $@ $^ 

%.o: %.cpp continuous.hpp Makefile
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm *~ *.o main main2

.PHONY: all clean run
