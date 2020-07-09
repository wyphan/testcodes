PROGRAM zgemmtest

  USE ISO_C_BINDING ! For c_sizeof and c_ptr

#ifdef _OPENMP
  USE OMP_LIB ! For OMP_GET_THREAD_NUM() and OMP_GET_NUM_THREADS()
#endif /* _OPENMP */

#ifdef USE_GPU

#ifdef _OPENACC

  USE cudafor ! for DEVICE, MANAGED attributes
  USE openacc

#else

  USE cuda_iface

#endif /* _OPENACC */

#ifdef CUBLAS
  USE cublas_iface
#endif /* CUBLAS */

#ifdef MAGMA
  USE magma_iface
#endif /* MAGMA */

#endif /* USE_GPU */

  IMPLICIT NONE

  ! Type constants
  INTEGER, PARAMETER :: dp = KIND(1.D0)
  INTEGER, PARAMETER :: dc = KIND((0.D0,1.D0))

  ! Constants
  INTEGER, PARAMETER :: kiB = 1024
  COMPLEX(KIND=dc), PARAMETER :: zzero = CMPLX( 0._dp, 0._dp )
  COMPLEX(KIND=dc), PARAMETER :: zone  = CMPLX( 1._dp, 0._dp )
  COMPLEX(KIND=dc), PARAMETER :: zimag = CMPLX( 0._dp, 1._dp )

  ! Matrix size
  INTEGER :: n

  ! Note: these aren't inside ifdefs since defaults will be set
  ! OpenMP variables
  INTEGER :: tid, nthreads, njobs, myidx

#ifdef USE_GPU

  ! Host copy of device pointers
  COMPLEX(KIND=dc), DIMENSION(:,:), ALLOCATABLE, TARGET :: d_A, d_B, d_C
  COMPLEX(KIND=dc) :: d_val

#ifdef USE_MANAGED
  ATTRIBUTES(managed) :: d_A, d_B, d_C
#endif /* USE_MANAGED */

#else

  ! Matrices
  COMPLEX(KIND=dc), DIMENSION(:,:), ALLOCATABLE :: A, B, C

  ! BLAS subroutine
  EXTERNAL :: zgemm

#endif /* USE_GPU */

  ! Local variables
  INTEGER :: i, j                             ! Loop indices
  COMPLEX(KIND=c_double_complex) :: dummy     ! for sizeof

  ! Timers
  INTEGER :: rate, t0, t1, t2
  REAL(KIND=dp) :: t01, t12, t01ave, t12ave

#ifdef NO_STIME
  ! Workaround for AOCC 2.1
  EXTERNAL :: my_clock_gettime
#else
  INTRINSIC :: SYSTEM_CLOCK
#endif /* NO_STIME */

#ifdef _OPENMP

  ! Note: The whole code is encased in one big PARALLEL directive
  !       PARAMETERS aren't affected by OpenMP directives, so no need to declare here
  !$OMP PARALLEL PRIVATE(tid,i,j,idx,njobs,dummy,rate,t0,t1,t2) &
  !$OMP          SHARED(nthreads,a,b,c,n,t01,t12,t01ave,t12ave)

  tid = OMP_GET_THREAD_NUM()
  nthreads = OMP_GET_NUM_THREADS()
  !WRITE(*,*) 'Thread ', tid, ' started'

#else

  tid = 0
  nthreads = 1

#endif /* _OPENMP */

  !$OMP MASTER

  WRITE(*,*) 'Using ', nthreads, ' threads'

  ! Read matrix size from stdin and echo it to stdout
  WRITE(*,*) 'Input matrix size'
  READ(*,'(I5)') n ! GCC doesn't like reading I0
  WRITE(*,*) n

  WRITE(*,*) 'Estimated memory usage:', &
             REAL( 3*n**2*C_SIZEOF(dummy), dp ) / REAL( kiB, dp )**2, &
             'MiB'

  ! Allocate matrices

#ifdef _OPENACC

  ! CPU arrays
  ALLOCATE( d_A(n,n) )
  ALLOCATE( d_B(n,n) )
  ALLOCATE( d_C(n,n) )

  ! Device arrays
  !$ACC ENTER DATA CREATE( d_a(1:n,1:n), d_b(1:n,1:n), d_c(1:n,1:n) )

#else

  ! CPU arrays
  ALLOCATE( A(n,n) )
  ALLOCATE( B(n,n) )
  ALLOCATE( C(n,n) )

#ifdef USE_GPU

  ! Device arrays
  CALL cudaMalloc_f( d_A, n*n )
  CALL cudaMalloc_f( d_B, n*n )
  CALL cudaMalloc_f( d_C, n*n )

#endif /* USE_GPU */

#endif /* _OPENACC */

  !$OMP END MASTER
  !$OMP BARRIER

#ifdef NO_STIME
  CALL my_clock_gettime( t0 )
#else
  ! Note: CPU_TIME is not threadsafe
  CALL SYSTEM_CLOCK( t0, COUNT_RATE=rate )
#endif /* NO_STIME */

#ifdef _OPENACC

  !$OMP MASTER

  ! Initialize matrices on device using OpenACC kernel
  WRITE(*,*) 'Initialize matrices on device'
  d_val = zone + zimag
  !$ACC KERNELS COPYIN( n, d_val ) PRESENT( d_a, d_b, d_c )
  !$ACC LOOP COLLAPSE(2)              
  DO j = 1, n
     DO i = 1, n
        d_a(i,j) = d_val
        d_b(i,j) = CONJG(d_val)
        d_c(i,j) = zzero
     END DO ! i
  END DO ! j
  !$ACC END LOOP
  !$ACC END KERNELS

  !$OMP MASTER

#else
  ! Initialize matrices on CPU

#ifdef _OPENMP

  ! Set up MPI-style work sharing, but for OpenMP
  njobs = n/nthreads  
  i = n - (nthreads*njobs)
  IF ( i/= 0 ) THEN
     IF ( tid < i ) THEN
        njobs = njobs + 1 ! Add extra jobs for lower tids
        myidx = tid*njobs + 1
     ELSE
        myidx = tid*njobs + i + 1 ! Offset start idx by remainder i
     END IF
  ELSE
     myidx = tid*njobs ! Everybody does the same number of jobs
  END IF

#else

  ! Note: if OpenMP is disabled, the following defaults are set:
  njobs = n
  myidx = 1

#endif /* _OPENMP */

  WRITE(*,*) 'Initialize matrices on CPU'
  !$OMP DO COLLAPSE(2)
  DO j = 0, njobs-1
     DO i = 1, n
        A(i,myidx+j) = zone + zimag
        B(i,myidx+j) = zone - zimag
        C(i,myidx+j) = zzero
     END DO ! i
  END DO ! j

#ifdef USE_GPU

  ! Not using OpenACC - matrices were initialized on CPU
  ! Now transfer them to device using CUDA
  !$OMP MASTER
  CALL cudaMemcpy_f( d_A, A, n*n, cudaMemcpyHostToDevice )
  CALL cudaMemcpy_f( d_B, B, n*n, cudaMemcpyHostToDevice )
  !$OMP END MASTER

#endif /* USE_GPU */

#endif /* _OPENACC */

#ifdef NO_STIME
  CALL my_clock_gettime( t1 )
#else
  CALL SYSTEM_CLOCK( t1 )
#endif /* NO_STIME */

#ifdef USE_GPU

  ! For now, only master is allowed to access the GPU
  !$OMP MASTER

#if defined(CUBLAS)

  ! Create cuBLAS handle
  CALL cublas_init_f

  ! Perform matrix multiply on device
  CALL cublas_zgemm_f( 'N', 'N', n, n, n, zone, d_A, n, d_B, n, zzero, d_C, n )

  ! Destroy cuBLAS handle
  CALL cublas_fin_f

#elif defined(MAGMA)

  ! Initialize MAGMA and create queue
  CALL magma_init_f

  ! Perform matrix multiply on device
  CALL magma_zgemm_f( 'N', 'N', n, n, n, zone, d_A, n, d_B, n, zzero, d_C, n )

  ! Destroy MAGMA queue and finalize MAGMA
  CALL magma_fin_f

#endif /* CUBLAS || MAGMA */

#ifdef _OPENACC

  ! Transfer data to CPU using OpenACC
  !$ACC UPDATE SELF( d_c )

#else

  ! Transfer data to CPU using CUDA
  CALL cudaMemcpy_f( C, d_C, n*n, cudaMemcpyDeviceToHost )

#endif /* _OPENACC */

  !$OMP END MASTER

#else

#ifdef _OPENMP

  ! Each thread performs matrix multiply on CPU for its own block of njobs
  CALL ZGEMM( 'N', 'N', njobs, n, n, zone, a(idx,1), n, b, n, zzero, c(idx,1), n)

#else

  ! Perform matrix multiply on 1 CPU
  CALL ZGEMM( 'N', 'N', n, n, n, zone, a, n, b, n, zzero, c, n)

#endif /* _OPENMP */

#endif /* USE_GPU */

#ifdef NO_STIME
  CALL my_clock_gettime( t2 )
#else
  CALL SYSTEM_CLOCK( t2 )
#endif /* NO_STIME */

  ! Validate result
#ifdef _OPENACC
  IF( INT(d_c(n,n)) /= 2*n ) THEN
     WRITE(*,*) 'Warning: incorrect result c_ij = ', INT(d_c(n,n)), &
#else
  IF( INT(c(n,n)) /= 2*n ) THEN
     WRITE(*,*) 'Warning: incorrect result c_ij = ', INT(c(n,n)), &
#endif /* _OPENACC */
                ', should be ', 2*n
  ELSE
     WRITE(*,*) 'OK'
  END IF

#ifdef _OPENMP

  ! Collect timings
  t01 = REAL( t1-t0, KIND=dp ) / REAL( rate, KIND=dp )
  t12 = REAL( t2-t1, KIND=dp ) / REAL( rate, KIND=dp )

  !$OMP MASTER
  t01ave = 0._dp
  t12ave = 0._dp
  !$OMP END MASTER

  !$OMP BARRIER

  !$OMP ATOMIC
    t01ave = t01ave + t01
  !$OMP ATOMIC
    t12ave = t12ave + t12

  !$OMP BARRIER

  !$OMP MASTER
  t01ave = t01ave / REAL( nthreads, KIND=dp )
  t12ave = t12ave / REAL( nthreads, KIND=dp )
  !$OMP END MASTER

#else

  t01ave = REAL( t1-t0, KIND=dp ) / REAL( rate, KIND=dp )
  t12ave = REAL( t2-t1, KIND=dp ) / REAL( rate, KIND=dp )

#endif /* _OPENMP */

  !$OMP MASTER

  ! Display timings
  WRITE(*,*) n, t01ave, t12ave

  ! Clean up

#ifdef _OPENACC

  !$ACC EXIT DATA IF( acc_is_present(d_a) ) DELETE( d_a )
  !$ACC EXIT DATA IF( acc_is_present(d_b) ) DELETE( d_b )
  !$ACC EXIT DATA IF( acc_is_present(d_c) ) DELETE( d_c )
  DEALLOCATE( d_a )
  DEALLOCATE( d_b )
  DEALLOCATE( d_c )

#else

#ifdef USE_GPU

  call cudaFree_f( d_A )
  call cudaFree_f( d_B )
  call cudaFree_f( d_C )

#endif /* USE_GPU */

  DEALLOCATE( a )
  DEALLOCATE( b )
  DEALLOCATE( c )

#endif /* _OPENACC */

  !$OMP END MASTER
  !$OMP END PARALLEL
  STOP

END PROGRAM zgemmtest
