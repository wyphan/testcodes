###############################################################################
# Compiler options
###############################################################################

CFLAGS = -g -O2
LDFLAGS =

###############################################################################
# Recipes that don't depend on GPU stack availability
###############################################################################
all build: checkapi detectgpu.x

clean:
	-rm -rf *.o

distclean: clean
	-rm -rf core *.x

###############################################################################
# Detect GPU API stack
###############################################################################
ifeq ("$(API)","")
checkapi:
	$(error Usage: make API=<API>)
else

# Standardize API name
supported_api_list := CUDA ROCM SYCL ACC OMP
cuda_str_list := cuda CUDA nv NV nvidia nVidia Nvidia NVIDIA
ifneq ($(findstring $(API),$(cuda_str_list)),)
selected_api := CUDA
else ifneq ($(findstring "$(API)","hip HIP rocm ROCM Rocm ROCm amd Amd AMD"),)
API := ROCM
else ifneq ($(findstring "$(API)","sycl SYCL dpcpp DPCPP dpc++ DPC++ intel Intel INTEL"),)
API := SYCL
else ifneq ($(findstring "$(API)","openacc OpenACC OPENACC acc ACC"),)
API := ACC
else ifneq ($(findstring "$(API)","openmp OpenMP OPENMP mp MP omp OMP"),)
API := OMP
else
checkapi:
	$(error Unsupported GPU API $(API), please choose from $(supported_api_list))
endif

endif

###############################################################################
# CUDA-specific options and targets
###############################################################################
ifeq ($(selected_api),CUDA)

ifeq ("${CUDA_HOME}","")
ifneq ("${NVHPC_CUDA_HOME}","")
CUDA_HOME := ${NVHPC_CUDA_HOME}
endif
endif
ifeq ("$(CUDA_HOME)","")
checkapi:
	$(error Please load a CUDA/NVHPC module or export CUDA_HOME to the appropriate path)
else
checkapi:
	@echo Using CUDA API at $(CUDA_HOME)
endif

NVCC := $(CUDA_HOME)/bin/nvcc
LDFLAGS += -L$(CUDA_HOME)/lib64 -lcudart
backend.o: backend_cuda.c
	$(NVCC) $(CFLAGS) -c $< -o $@

endif

###############################################################################
# ROCm-specific options and targets
###############################################################################
ifeq ("$(API)",ROCM)

ifeq ("${ROCM_PATH}","")
checkapi:
	$(error Please load a ROCm module or export ROCM_PATH to the appropriate path)
else
checkapi:
	@echo Using ROCm API at $(ROCM_PATH)
endif

HIPCC := $(ROCM_PATH)/bin/hipcc
LDFLAGS += -L$(ROCM_PATH)/lib -lamdhip64
backend.o: backend_rocm.c
	$(HIPCC) $(CFLAGS) -c $< -o $@

endif

###############################################################################
# SYCL-specific options and targets
###############################################################################
ifeq ("$(API)",SYCL)

endif

###############################################################################
# OpenACC-specific options and targets
###############################################################################
ifeq ("$(API)",ACC)

endif

###############################################################################
# OpenACC-specific options and targets
###############################################################################
ifeq ("$(API)",OMP)

endif

###############################################################################
# Host-side targets
###############################################################################
detectgpu.x: main.o common_mpi.o backend.o
	$(MPICC) -o $@ $^ $(LDFLAGS)

common_mpi.o: common_mpi.c
	$(MPICC) $(CFLAGS) -c $< -o $@

main.o: main.c
	$(MPICC) $(CFLAGS) -c $< -o $@
