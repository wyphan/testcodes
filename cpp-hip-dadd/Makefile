###############################################################################
# Recipes that don't depend on ROCm availability
###############################################################################

build: checkrocm dadd.x

clean:
	-rm -rf *.x
	-rm -rf core

distclean: clean
	-rm -rf hpctoolkit-* *.m *.d

###############################################################################
# Detect ROCm installation
###############################################################################
checkrocm:
ifeq ("${ROCM_PATH}","")
	$(error Please load a ROCm module or export ROCM_PATH to the appropriate path)
endif

###############################################################################
# Compiler options
###############################################################################

HIPCC := ${ROCM_PATH}/bin/hipcc
CFLAGS = -g -O2
LDFLAGS =

truthy = 1 on ON yes YES true TRUE
ifdef USE_ROCTX
ifneq ($(findstring $(USE_ROCTX),$(truthy)),)
CFLAGS += -DUSE_ROCTX
LDFLAGS += -L${ROCM_PATH}/lib -lroctx64
endif
endif

dadd.x: dadd.hip.cpp
	$(HIPCC) $(CFLAGS) $< $(LDFLAGS) -o $@
