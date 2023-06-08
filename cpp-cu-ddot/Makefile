###############################################################################
# Recipes that don't depend on ROCm availability
###############################################################################

build: ddot.x

clean:
	-rm -rf *.x
	-rm -rf core

distclean: clean
	-rm -rf hpctoolkit-* *.m *.d

###############################################################################
# Detect ROCm installation
###############################################################################
ifeq ("${CUDA_HOME}","")
ifneq ("${NVHPC_CUDA_HOME}","")
CUDA_HOME := ${NVHPC_CUDA_HOME}
else
$(error Please load a CUDA module or export CUDA_HOME to the appropriate path)
endif
endif

###############################################################################
# Compiler options
###############################################################################

NVCC := $(CUDA_HOME)/bin/nvcc
CFLAGS = -g -O2 -x cu
LDFLAGS =

truthy = 1 on ON yes YES true TRUE
ifdef USE_NVTX
ifneq ($(findstring $(USE_NVTX),$(truthy)),)
CFLAGS += -DUSE_NVTX
LDFLAGS += -L$(CUDA_HOME)/lib64 -lnvToolsExt
endif
endif

ddot.x: ddot.cuda.cpp
	$(NVCC) $(CFLAGS) $< $(LDFLAGS) -o $@
