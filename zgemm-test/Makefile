#==============================================================================
# Compiler options
#==============================================================================

# GCC
#gccopts = -g2 -Ofast -cpp -fopenmp -fopt-info-optall

# IBM XL
xlfopts = -g2 -O2 -qpreprocess -qipa -qarch=pwr9 -qtune=pwr9:smt4 -qunroll -qessl -qsmp=omp:auto -qreport=smplist

# PGI
#pgfopts       = -g -O0 -Minform=warn -traceback
#pgfopts       = -gopt -O -Mpreprocess -Munroll -Mlre -Mvect=simd -Mflushz -Mcache_align -Mnoinline -Mconcur -Minfo=ftn,opt,loop,lre,vect,par,acc -traceback
pgfopts       = -gopt -fast -Mpreprocess -Minfo=ftn,opt,loop,lre,vect,par,acc -traceback
pgfaccopts    = -acc -ta=tesla:lineinfo -Mcuda
pgfcudalink   = -Mcudalib=cublas -cudalibs

# LLVM
#flangopts=-DNO_STIME -g -Ofast -Mfreeform -Mpreprocess -mp
#clangopts=-g -Ofast -fopenmp
#flanglink=-mp

#==============================================================================
# Libraries
#==============================================================================

# IBM ESSL
essllink=-L${OLCF_ESSL_ROOT}/lib64 -lesslsmp

# IBM XL support libs for ESSL
xlfdir=/sw/summit/xl/16.1.1-5/xlf/16.1.1
xlsmpdir=/sw/summit/xl/16.1.1-5/xlsmp/5.1.1
xlflink=-L$(xlfdir)/lib -lxlf90_r
xlsmplink=-L$(xlsmpdir)/lib -lxlsmp

# AMD AOCL
aocldir=/opt/amd/aocl/2.1
aocllink=-L$(aocldir)/libs -lblis-mt

# OpenBLAS
oblasdir=/opt/openblas/openblas-0.3-9/pgi-19.10+nollvm
oblaslink=-L$(oblasdir)/lib -lopenblas

# cuBLAS
cublasopts = -DUSE_GPU -DCUBLAS

# MAGMA
magmaopts  = -DUSE_GPU -DMAGMA
magmadir=${PROJWORK}/mat201/magma-2.5.3/pgi-20.1+cuda-10.1+essl
#magmadir=/opt/magma/magma-2.5.3/pgi-19.10+nollvm/cuda-10.2+openblas-0.3.9
magmainc=-I$(magmadir)/include
magmalink=-L$(magmadir)/lib $(magmadir)/lib/magma2*.o -lmagma

#==============================================================================

.PHONY:

ibm:
	export CC=xlc_r; export FC=xlf90_r; make ibmessl

ibmessl:
	$(FC) $(xlfopts) zgemmtest.F90 -o zgemm-xlf-essl $(essllink) $(xlsmplink)

pgi:
	export CC=pgcc; export FC=pgfortran; make pgibuiltin pgiessl pgicublas pgimagma

pgibuiltin:
	$(FC) $(pgfopts) zgemmtest.F90 -o zgemm-pgi-builtin -lblas

pgiessl:
	$(FC) $(pgfopts) zgemmtest.F90 -o zgemm-pgi-essl $(essllink) $(xlflink) $(xlsmplink)

pgicublas:
	$(FC) $(pgfopts) $(pgfaccopts) $(cublasopts) -c mod_cublas_iface.F90 -o cublas_iface_f.o
	$(FC) $(pgfopts) $(pgfaccopts) $(cublasopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(pgfopts) $(pgfaccopts) cublas_iface_f.o zgemmtest.o -o zgemm-pgi-cublas $(pgfcudalink)

pgimagma:
	$(FC) $(pgfopts) $(pgfaccopts) $(magmainc) $(magmaopts) -c mod_magma_iface.F90 -o magma_iface.o
	$(FC) $(pgfopts) $(pgfaccopts) $(magmainc) $(magmaopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(pgfopts) $(pgfaccopts) magma_iface.o zgemmtest.o -o zgemm-pgi-magma $(magmalink) $(pgfcudalink)

llvm:
	export CC=clang; export FC=flang; make flangaocl

flangaocl: llvm
	$(CC) $(clangopts) -c clock_gettime.c -o clock_gettime.o
	$(FC) $(flangopts) -c zgemmtest.F90 -o zgemmtest.o
	$(FC) $(flanglink) zgemmtest.o clock_gettime.o -o zgemm-flang-aocl $(aocllink)

clean:
	-rm *.o *.mod
	-[[ -e zgemm-xlf-essl    ]] && rm zgemm-xlf-essl
	-[[ -e zgemm-pgi-builtin ]] && rm zgemm-pgi-builtin
	-[[ -e zgemm-pgi-essl    ]] && rm zgemm-pgi-essl
	-[[ -e zgemm-pgi-cublas  ]] && rm zgemm-pgi-cublas
	-[[ -e zgemm-pgi-magma   ]] && rm zgemm-pgi-magma
	-[[ -e zgemm-flang-aocl  ]] && rm zgemm-flang-aocl
