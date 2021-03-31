#==============================================================================
# Compiler options
#==============================================================================

# GCC
gccopts  = -cpp -g2 -Ofast -fopenacc -foffload=amdgcn-amdhsa="-march=gfx900"

# NVHPC
nvopts   = -Mpreprocess -gopt -fast -acc=gpu -gpu=managed -Minfo=accel -traceback

#==============================================================================
# Libraries
#==============================================================================

# AMD AOCL
aocldir     = /opt/AMD/aocl/aocl-linux-aocc-2.2.0
aocllink    = -L$(aocldir)/libs -lblis-mt

# Intel MKL
mklincgcc   = -m64  -I"${MKLROOT}/include"
mkllinkgcc  = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
              -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
mklincnv    = -I"${MKLROOT}/include"
mkllinknv   = -L${MKLROOT}/lib/intel64 \
              -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# Reference BLAS
refblasdir  = /opt/reflapack/lapack-3.9.0/nv-20.11
refblaslink = -L$(refblasdir)/lib -lblas

# OpenBLAS
oblasdir    = /opt/openblas/openblas-0.3.10/gcc-9.3.0
oblaslink   = -L$(oblasdir)/lib -lopenblas

# MAGMA
#magmadir    = /opt/magma/magma-2.5.3/gcc-9.3.0+cuda11.0+openblas0.3.10
magmadir    = /opt/magma/magma-2.5.4/nv20.11+cuda11.2+reflapack3.9.0
#magmadir    = /opt/magma/hipmagma-2.0.0/gcc-10.2.0+rocm3.10.0+openblas0.3.10
magmainc    = -I$(magmadir)/include
magmalink   = -L$(magmadir)/lib -lmagma

#==============================================================================
# Pick your combination here
#==============================================================================

#COMPILER = gcc
COMPILER = nv

FC = mpif90

# AMD AOCL, MAGMA
#LINK = $(aocllink) $(magmalink)

# Intel MKL, MAGMA (with GCC)
#INC  = $(mklincgcc)
#LINK = $(mkllinkgcc) $(magmalink)

# Intel MKL, MAGMA (with PGI/NVHPC)
#INC  = $(mklincnv)
#LINK = $(mkllinknv) $(magmalink)

# Reference BLAS, MAGMA
LINK = $(refblaslink) $(magmalink)

# OpenBLAS, MAGMA
#LINK = $(oblaslink) $(magmalink)

#==============================================================================

modules = 

.PHONY: main

%.o: %.mod

.SUFFIXES: .f90
.f90.o:
ifeq ($(COMPILER),gcc)
	$(FC) $(INC) $(gccopts) -c -o $@ $<
else ifeq ($(COMPILER),nv)
	$(FC) $(INC) $(nvopts) -c -o $@ $<
else
	$(FC) $(INC) ${FCOPTS} -c -o $@ $<
endif

main: mod_prec.o mod_blas.o mod_magma.o main.o
ifeq ($(COMPILER),gcc)
	$(FC) $(gccopts) *.o $(LINK) -o magmatest-gcc.x
else ifeq ($(COMPILER),nv)
	$(FC) $(nvopts) *.o $(LINK) -o magmatest-nv.x
else
	$(FC) ${FCOPTS} *.o $(LINK) -o magmatest-$(COMPILER).x
endif


clean:
	-rm *.o *.mod

distclean: clean
	-rm *.x

