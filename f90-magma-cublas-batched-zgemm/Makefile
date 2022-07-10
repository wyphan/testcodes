#==============================================================================
# Compiler options
#==============================================================================

# PGI
pgiopts  = -DADD_ -gopt -O3 -traceback
pgiacc   = -acc=autopar -ta=tesla:cc70 -Minfo=accel

# NVHPC
nvopts   = -DADD_ -gopt -O3 -traceback
nvacc    = -acc=gpu -acc=autopar -gpu=cc70 -Minfo=accel

#==============================================================================
# Libraries
#==============================================================================

# cuBLAS
cudadir  = ${OLCF_CUDA_ROOT}
cudainc  = -I$(cudadir)/include
cudalink = -L$(cudadir)/lib64 -lcudart -lcublas

# IBM ESSL
essldir  = ${OLCF_ESSL_ROOT}
essllink = -L$(essldir)/lib64 -lessl

# Reference LAPACK
reflapackdir  = ${OLCF_NETLIB_LAPACK_ROOT}
reflapacklink = -L$(reflapackdir)/lib64 -llapack

# MAGMA
#magmadir  = /ccs/proj/mat201/magma-2.5.4/pgi-20.4+cuda-10.2.89+essl
magmadir  = /ccs/proj/mat201/magma-2.5.4/nv-21.3+cuda-10.2.89+essl
magmainc  = -I$(magmadir)/include
magmalink = -L$(magmadir)/lib -lmagma

#==============================================================================
# Pick your combination here
#==============================================================================

#COMPILER = pgi
COMPILER = nv

ifeq ($(COMPILER),pgi)
CXX = pgc++
CXXOPTS = $(pgiopts)
ACCOPTS = $(pgiacc)
else ifeq ($(COMPILER),nv)
CXX = nvc++
CXXOPTS = $(nvopts)
ACCOPTS = $(nvacc)
endif

INC = $(cudainc) $(magmainc)
LIBS = $(magmalink) $(essllink) $(reflapacklink) $(essllink) $(cudalink)

#==============================================================================

.PHONY: all

.SUFFIXES: .cpp

.cpp.o:
	$(CXX) $(INC) $(CXXOPTS) $(ACCOPTS) -c -o $@ $<

all: magma_util.o wigner_d_matrix_acc.o testing_zgemm_batched.o
	$(CXX) $(CXXOPTS) $(ACCOPTS) *.o -o testing_zgemm_batched-$(COMPILER).x $(LIBS)

clean:
	-rm *.o

distclean: clean
	-rm *.x
