# -*- mode: makefile-gmake -*-

###############################################################################
# Utility functions for parsing lists
###############################################################################

# List delimiters
null :=
space := $(null) #
comma := ,

# Functions to convert between list delimiters
csl = $(subst $(space),$(comma),$(strip $(1)))
ssl = $(subst $(comma),$(space),$(strip $(1)))

###############################################################################
# Recipes that don't depend on ROCm availability
###############################################################################

CCNAME = $(notdir $(CC))

build: checkgpuarch ddot.$(CCNAME).x

clean:
	-rm -rf *.x

distclean: clean
	-rm -rf core
	-rm -rf hpctoolkit-* *.m *.d

ifeq ("$(ARCH)","")
checkgpuarch:
	$(error Usage: make ARCH=<comma-separated_list_of_gpu_arches>)
else
checkgpuarch:
	@echo ARCH = $(ARCH)
endif

###############################################################################
# OpenMP compiler flags
###############################################################################

ifeq ("gcc","$(findstring gcc,$(CCNAME))")

  OMPFLAGS = -fopenmp
  ARCHLIST = $(call ssl,$(ARCH))
  ifeq ("sm","$(findstring sm,$(ARCH))")
    ARCHFLAGS = $(foreach arch,$(ARCHLIST),-foffload=nvptx-none="-misa=$(arch)")
  else ifeq ("gfx","$(findstring gfx,$(ARCH))")
    ARCHFLAGS = $(foreach arch,$(ARCHLIST),-foffload=amdgcn-amdhsa="-march=$(arch)")
  endif
  OMPTGTFLAGS = -foffload-options="-g -O2" $(ARCHFLAGS)

else ifeq ("nvc","$(findstring nvc,$(CCNAME))")

  OMPFLAGS = -mp 
  ARCHFLAGS = -gpu=$(ARCH)
  OMPTGTFLAGS = -mp=gpu $(ARCHFLAGS)

else ifeq ("amdclang","$(findstring amdclang,$(CCNAME))")

  OMPFLAGS = -fopenmp
  ARCHLIST = $(call ssl,$(ARCH))
  ARCHFLAGS = $(foreach arch,$(ARCHLIST),--offload-arch=$(arch))
  OMPTGTFLAGS = $(ARCHFLAGS)

else ifeq ("icx","$(findstring icx,$(CCNAME))")

  OMPFLAGS = -qopenmp
  ARCHFLAGS = -fopenmp-targets=spir64
  OMPTGTFLAGS = $(ARCHFLAGS)

endif

###############################################################################
# Compiler options
###############################################################################

CFLAGS = -g -O2
LDFLAGS =

ddot.$(CCNAME).x: ddot.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OMPTGTFLAGS) $< $(LDFLAGS) -o $@
