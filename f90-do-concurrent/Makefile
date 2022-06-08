###############################################################################
# Compiler options
###############################################################################

# NVIDIA nvfortran
COMPILER = nv
FC = nvfortran
FCFLAGS = -stdpar -Minfo=accel
LDFLAGS = -stdpar

###############################################################################
# Executable name
###############################################################################

EXE = stdpar.$(COMPILER).x

###############################################################################
# Prefix rules
###############################################################################

.PREFIXES: .f90

# Prevent invocation of m2c (Modula-2 compiler)
%.o: %.mod

%.f90.o: %.f90
	$(FC) $(FCFLAGS) -c $< -o $*.o

###############################################################################
# Phony targets
.PHONY: all

all: main

# Build main
main: stdpar.f90.o
	$(FC) stdpar.o -o $(EXE) $(LDFLAGS)

# Clean up
clean:
	-rm -f stdpar.o
	-rm -f $(EXE)
