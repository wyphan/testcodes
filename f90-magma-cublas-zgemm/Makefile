#==============================================================================
# Compiler options
#==============================================================================

# GCC
gccopts = -g2 -Ofast -cpp -fopenmp -fopt-info-optall

# IBM XL
xlfopts = -g2 -O2 -qpreprocess -qipa -qarch=pwr9 -qtune=pwr9:smt4 -qunroll -qessl -qsmp=omp:auto -qreport=smplist

# PGI / NVIDIA HPC SDK
#pgfopts     = -g -O0 -Minform=warn -traceback
#pgfopts     = -gopt -O -Mpreprocess -Munroll -Mlre -Mvect=simd -Mflushz -Mcache_align -Mnoinline -Mconcur -Minfo=ftn,opt,loop,lre,vect,par,acc -traceback
pgfopts     = -gopt -fast -Mpreprocess -Minfo=ftn,opt,loop,lre,vect,par,acc -traceback
pgfaccopts  = -acc -ta=tesla:lineinfo -Mcuda
pgfcudalink = -Mcudalib=cublas -cudalibs

# AMD AOCC
flangopts = -g -Ofast -Mfreeform -Mpreprocess -mp
clangopts = -g -Ofast -fopenmp
flanglink = -mp

#==============================================================================
# IBM ESSL
#==============================================================================

# IBM XL support libs for ESSL (Summit)
xlfdir    = /sw/summit/xl/16.1.1-5/xlf/16.1.1
xlsmpdir  = /sw/summit/xl/16.1.1-5/xlsmp/5.1.1

# IBM ESSL (Summit)
essllink  = -L${OLCF_ESSL_ROOT}/lib64 -lesslsmp
xlflink   = -L$(xlfdir)/lib -lxlf90_r
xlsmplink = -L$(xlsmpdir)/lib -lxlsmp

# IBM XL support libs for ESSL (Tellico)
#xlfdir    = ${ESSL_DIR}
#xlsmpdir  = ${ESSL_DIR}

# IBM ESSL (Tellico)
#essllink  = -L${ESSL_DIR}/lib64 -lesslsmp
#xlflink   = -L$(xlfdir)/lib64 -lxlf90_r
#xlsmplink = -L$(xlsmpdir)/lib64 -lxlsmp

#==============================================================================
# AMD AOCL
#==============================================================================

# BaseCamp (AOCC)
aocldir  = /opt/AMD/aocl/aocl-linux-aocc-3.2.0
aocllink = -L$(aocldir)/libs -lblis-mt

#==============================================================================
# OpenBLAS
#==============================================================================

# BaseCamp (PGI)
#oblasdir  = /opt/openblas/openblas-0.3-9/pgi-19.10+nollvm

# BaseCamp (NVHPC)
oblasdir  = /opt/openblas/openblas-0.3-13/nv-20.11

oblaslink =-L$(oblasdir)/lib -lopenblas

#==============================================================================
# cuBLAS
#==============================================================================

cublasopts = -DUSE_GPU -DCUBLAS

#==============================================================================
# MAGMA
#==============================================================================

# Summit (PGI)
magmadir   = ${PROJWORK}/mat201/magma-2.5.3/pgi-20.1+cuda-10.1+essl

# Tellico (NVHPC)
#magmadir   = ${HOME}/nvlibs/nv-20.9/magma-2.5.4+cuda10.1+essl

# BaseCamp (PGI)
#magmadir   = /opt/magma/magma-2.5.3/pgi-19.10+nollvm/cuda-10.2+openblas-0.3.9

# BaseCamp (NVHPC)
#magmadir   = /opt/magma/magma-2.5.4/nv20.11+cuda-11.2+openblas-0.3.13

magmaopts  = -DUSE_GPU -DMAGMA
magmainc   = -I$(magmadir)/include
magmalink  = -L$(magmadir)/lib -lmagma

#==============================================================================

.PHONY:

ibm:
	export CC=xlc_r; export FC=xlf90_r

ibm-all: ibmessl

ibmessl: ibm
	$(FC) $(xlfopts) zgemmtest.F90 -o zgemm-xlf-essl.x $(essllink) $(xlsmplink)

pgi:
	export CC=pgcc; export FC=pgfortran

pgi-all: pgibuiltin pgiessl pgicublas pgimagma

pgibuiltin: pgi
	$(FC) $(pgfopts) zgemmtest.F90 -o zgemm-pgi-builtin.x -lblas

pgiessl: pgi
	$(FC) $(pgfopts) zgemmtest.F90 -o zgemm-pgi-essl.x $(essllink) $(xlsmplink)

pgicublas: pgi
	$(FC) $(pgfopts) $(pgfaccopts) $(cublasopts) -c mod_cublas_iface.F90 -o cublas_iface_f.o
	$(FC) $(pgfopts) $(pgfaccopts) $(cublasopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(pgfopts) $(pgfaccopts) cublas_iface_f.o zgemmtest.o -o zgemm-pgi-cublas.x $(pgfcudalink)

pgimagma: pgi
	$(FC) $(pgfopts) $(pgfaccopts) $(magmainc) $(magmaopts) -c mod_magma_iface.F90 -o magma_iface.o
	$(FC) $(pgfopts) $(pgfaccopts) $(magmainc) $(magmaopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(pgfopts) $(pgfaccopts) magma_iface.o zgemmtest.o -o zgemm-pgi-magma.x $(magmalink) $(pgfcudalink)

nv:
	export CC=nvc; export FC=nvfortran

nv-all: nvbuiltin nvessl nvcublas nvmagma

nvbuiltin: nv
	$(FC) $(pgfopts) zgemmtest.F90 -o zgemm-nv-builtin.x -lblas

nvessl: nv
	$(FC) $(pgfopts) zgemmtest.F90 -o zgemm-nv-essl.x $(essllink) $(xlsmplink)

nvcublas: nv
	$(FC) $(pgfopts) $(pgfaccopts) $(cublasopts) -c mod_cublas_iface.F90 -o cublas_iface_f.o
	$(FC) $(pgfopts) $(pgfaccopts) $(cublasopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(pgfopts) $(pgfaccopts) cublas_iface_f.o zgemmtest.o -o zgemm-nv-cublas.x $(pgfcudalink)

nvmagma: nv
	$(FC) $(pgfopts) $(pgfaccopts) $(magmainc) $(magmaopts) -c mod_magma_iface.F90 -o magma_iface.o
	$(FC) $(pgfopts) $(pgfaccopts) $(magmainc) $(magmaopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(pgfopts) $(pgfaccopts) magma_iface.o zgemmtest.o -o zgemm-nv-magma.x $(magmalink) $(pgfcudalink)

aocc:
	export CC=clang; export FC=flang

aocc-all: flangaocl

flangaocl:
	$(FC) $(flangopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(flanglink) zgemmtest.o -o zgemm-flang-aocl.x $(aocllink)

clean:
	-rm *.o *.mod *.x
