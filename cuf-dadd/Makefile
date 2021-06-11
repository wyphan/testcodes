#==============================================================================
# Compiler options
#==============================================================================

# PGI
pgicuda = -Mcuda
pgiopts  = -Mpreprocess -gopt -fast -Minfo=all -traceback

# NVHPC
nvcuda = -Mcuda
nvopts   = -Mpreprocess -gopt -fast -Mcuda -Minfo=all -traceback

#==============================================================================
# Pick your combination here
#==============================================================================

#COMPILER = pgi
COMPILER = nv

ifeq ($(COMPILER),pgi)
FC = pgfortran
FCOPTS = $(pgiopts)
CUDAOPTS = $(pgicuda)
else ifeq ($(COMPILER),nv)
FC = nvfortran
FCOPTS = $(nvopts)
CUDAOPTS = $(nvcuda)
endif

#==============================================================================

.PHONY: main

%.o: %.mod

.SUFFIXES: .f90 .cuf

.f90.o:
	$(FC) $(FCOPTS) -c -o $@ $<

.cuf.o:
	$(FC) $(FCOPTS) $(CUDAOPTS) -c -o $@ $<

main: mod_prec.o kernels.o
	$(FC) $(FCOPTS) $(CUDAOPTS) -c -o main.o main.f90
	$(FC) $(FCOPTS) *.o -o cuf-dadd-$(COMPILER).x

clean:
	-rm *.o *.mod

distclean: clean
	-rm *.x
