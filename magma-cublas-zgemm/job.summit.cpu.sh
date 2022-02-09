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

# Load modules
module load job-step-viewer
module load xl
module load essl

# Prepare the job
echo "`date` Job ${LSB_JOBID} launched from `hostname`"
cd ${LS_SUBCWD}
echo "Workdir is `pwd`"

export jsopts="-r 1 -c 42 -a 1 -g ${gpures} -E OMP_NUM_THREADS=$(( 42 * ${smtlv} )) -brs"

echo "`date` Launching first job (IBM+ESSL) with 1 rank, $(( 42 * ${smtlv} )) threads, ${gpures} GPU"
jsrun ${jsopts} echo "16384" | ./zgemm-xlf-essl > ibm-essl.txt

module load pgi
module load essl

echo "`date` Launching second job (PGI+builtin) with 1 rank, $(( 42 * ${smtlv} )) threads, ${gpures} GPU"
jsrun ${jsopts} echo "16384" | ./zgemm-pgi-builtin > pgi-builtin.txt

echo "`date` Launching third job (PGI+ESSL) with 1 rank, $(( 42 * ${smtlv} )) threads, ${gpures} GPU"
jsrun ${jsopts} echo "16384" | ./zgemm-pgi-essl > pgi-essl.txt

echo "`date` Done"
