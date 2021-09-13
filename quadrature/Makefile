# Compiler options
COMPILER = gcc
CXX = g++
CXXFLAGS = -g -O2 -fopt-info -march=native -mtune=native
LDFLAGS = -lm

# Executable name
EXE = quad.$(COMPILER).x

# Prefix rules

.PREFIXES: .cpp

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $*.o

# List sources and targets
HDR = quadrature.hpp quad_simpson.hpp
SRC = quadrature.cpp quad_simpson.cpp

OBJ = main.o $(SRC:.cpp=.o)
TGT = $(SRC:.cpp=.cpp.o)


# Phony targets
.PHONY: all

all: main

# Build main
main: quadrature.cpp.o quad_simpson.cpp.o
	$(CXX) $(OBJ) -o $(EXE) $(LDFLAGS)

# Clean up
clean:
	-rm -f $(OBJ)
