
#!/bin/bash

#BSUB -P MAT201
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -alloc_flags "smt4"
#BSUB -J zgemm_test
#BSUB -N wphan@vols.utk.edu

# Make sure this matches the bsub alloc_flags!
# Number of OpenMP threads per physical core
export smtlv=4

# Number of GPU per resource set (0 or 1)
export gpures=0

# Matrix size
export n=16384

# Load modules
module load job-step-viewer
module load pgi/20.1
module load essl

# Prepare the job
echo "`date` Job ${LSB_JOBID} launched from `hostname`"
cd ${LS_SUBCWD}
echo "Workdir is `pwd`"

export_jsopts() {
  export jsopts="-r 1 -c 42 -a 1 -g ${gpures} -E OMP_NUM_THREADS=$(( 42 * ${smtlv} )) -brs"
}
export_jsopts

echo "`date` Launching first job (PGI+ESSL) with 1 rank, $(( 42 * ${smtlv} )) threads, ${gpures} GPU"
jsrun ${jsopts} echo "$n" | ./zgemm-pgi-essl > pgi-essl.txt
echo "`date` First job done"

# Load CUDA module
module load cuda

export gpures=1
export_jsopts

echo "`date` Launching second job (PGI+OpenACC+cuBLAS) with 1 rank, $(( 42 * ${smtlv} )) threads, ${gpures} GPU"
jsrun ${jsopts} echo "$n" | ./zgemm-pgi-cublas > pgi-cublas.txt
echo "`date` Second job done"

# Load modules for MAGMA and export LD_LIBRARY_PATH
module load netlib-lapack
export MAGMADIR=/gpfs/alpine/proj-shared/mat201/magma-2.5.3/pgi-20.1+cuda-10.1+essl
export LD_LIBRARY_PATH=${MAGMADIR}/lib:${LD_LIBRARY_PATH}

echo "`date` Launching third job (PGI+OpenACC+MAGMA) with 1 rank, $(( 42 * ${smtlv} )) threads, ${gpures} GPU"
jsrun ${jsopts} -E LD_LIBRARY_PATH echo "$n" | ./zgemm-pgi-magma > pgi-magma.txt
echo "`date` Third job done"
