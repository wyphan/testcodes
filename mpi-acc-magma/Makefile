#==============================================================================
# Compiler options
#==============================================================================

# GCC
gccopts = -g2 -Ofast -fopenacc -foffload=amdgcn-amdhsa="-march=gfx900"

# NVHPC
nvopts = -gopt -fast -acc=gpu -Minfo=accel -traceback

#==============================================================================
# Libraries
#==============================================================================

# AMD AOCL
aocldir  = /opt/AMD/aocl/aocl-linux-aocc-2.2.0
aocllink = -L$(aocldir)/libs -lblis-mt

# Intel MKL
mklincgcc  = -m64  -I"${MKLROOT}/include"
mkllinkgcc = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
             -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
mklincnv   = -I"${MKLROOT}/include"
mkllinknv  = -L${MKLROOT}/lib/intel64 \
             -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# OpenBLAS
oblasdir  = /opt/openblas/openblas-0.3.10/gcc-9.3.0
oblaslink = -L$(oblasdir)/lib -lopenblas

# MAGMA
#magmadir  = /opt/magma/magma-2.5.3/gcc-9.3.0+cuda11.0+openblas0.3.10
magmadir  = /opt/magma/magma-2.5.4/nv20.11+cuda11.2+reflapack3.9.0
#magmadir  = /opt/magma/hipmagma-2.0.0/aomp11.8+rocm3.7.0+openblas0.3.10
magmainc  = -I$(magmadir)/include
magmalink = -L$(magmadir)/lib -lmagma

#==============================================================================
# Pick your combination here
#==============================================================================

# AMD AOCL, MAGMA
#LINK    = $(aocllink) $(magmalink)

# Intel MKL, MAGMA (with GCC)
#F90_INC = $(mklincgcc)
#LINK    = $(mkllinkgcc) $(magmalink)

# Intel MKL, MAGMA (with PGI/NVHPC)
#F90_INC = $(mklincnv)
#LINK    = $(mkllinknv) $(magmalink)

# OpenBLAS, MAGMA
LINK    = $(oblaslink) $(magmalink)

#==============================================================================

modules = mod_prec.f90 mod_blas.f90 mod_magma.f90

.PHONY:

.SUFFIXES: .f90

%.o: %.mod
.f90.o:
	$(FC) $(F90_OPTS) -c -o $*.o $<

mod_blas.o: mod_blas.f90
	$(FC) $(F90_INC) $(F90_OPTS) main.f90 -o magmatest-gcc.x

gcc: modules
	export FC=mpif90; export F90_OPTS=$(gccopts)
	$(FC) $(F90_OPTS) main.f90 -o magmatest-gcc.x

nvhpc: modules
	export FC=mpif90; export F90_OPTS=$(nvopts)
	$(FC) $(F90_OPTS) main.f90 -o magmatest-nv.x

clean:
	-rm *.o *.mod *.x
