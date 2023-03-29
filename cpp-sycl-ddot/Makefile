###############################################################################
# Compiler options
###############################################################################

# Intel
CXX = icpx
CXXFLAGS = -g -O2 -march=native -mtune=native -fsycl -fsycl-targets=spir64
LDFLAGS = -fsycl

# hipSYCL/Open SYCL
#CXX = syclcc
#CXXFLAGS = -g -O2 --hipsycl-platform=rocm --hipsycl-gpu-arch=gfx900
#LDFLAGS =

###############################################################################
# Executable name
###############################################################################

EXE = ddot.$(CXX).x

###############################################################################
# Prefix rules
###############################################################################

.PREFIXES: .cpp

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $*.o

###############################################################################
# Phony targets
.PHONY: all

all: main

SRC = ddot.sycl.cpp
OBJ = $(SRC:.cpp=.o)
TGT = $(SRC:.cpp=.cpp.o)

# Build main
main: $(TGT)
	$(CXX) $(OBJ) -o $(EXE) $(LDFLAGS)

# Clean up
clean:
	-rm -f $(OBJ)

distclean: clean
	-rm -f $(EXE)
