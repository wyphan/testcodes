#==============================================================================
# Compiler options
#==============================================================================

# GCC
gccopts  = -g2 -Ofast

# PGI
pgiopts  = -gopt -fast -traceback

# NVHPC
nvopts   = -gopt -fast -traceback

#==============================================================================
# Libraries
#==============================================================================

# HDF5
#hdf5inc  = -I/opt/gcc9libs/include
#hdf5link = -L/opt/gcc9libs/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -ldl
#hdf5inc  = -I/opt/pgilibs/include
#hdf5link = -L/opt/pgilibs/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -ldl
#hdf5inc  = -I/opt/nv20libs/include
#hdf5link = -L/opt/nv20libs/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -ldl
hdf5inc  = -I${OLCF_HDF5_ROOT}/include
hdf5link = -L${OLCF_HDF5_ROOT}/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -ldl

#==============================================================================
# Pick your combination here
#==============================================================================

#COMPILER = gcc
#COMPILER = pgi
COMPILER = nv

ifeq ($(COMPILER),gcc)
FC = gfortran
FCOPTS = $(gccopts)
else ifeq ($(COMPILER),pgi)
FC = pgf90
FCOPTS = $(pgiopts)
else ifeq ($(COMPILER),nv)
FC = nvfortran
FCOPTS = $(nvopts)
endif

#==============================================================================

.PHONY: main

%.o: %.mod

.SUFFIXES: .f90
.f90.o:
	$(FC) $(hdf5inc) $(FCOPTS) -c -o $@ $<

main: mod_prec.o mod_hdf5.o main.o
	$(FC) $(FCOPTS) *.o $(hdf5link) -o hdf5test.x

clean:
	-rm *.o *.mod

distclean: clean
	-rm *.x

