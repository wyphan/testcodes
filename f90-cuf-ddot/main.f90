PROGRAM cuf_dadd

  USE mod_prec

#if defined(__PGI) || defined(__NVCOMPILER)
  USE cudafor
#else
  USE ISO_C_BINDING
  USE mod_cudart
#endif /* __PGI || __NVCOMPILER */

#ifdef _CUDA
  USE cuf_kernels
#endif

  IMPLICIT NONE

  ! Compiler identification
#ifdef __GFORTRAN__
  CHARACTER(LEN=*), PARAMETER :: compiler = 'GCC GFortran'
#endif /* __GFORTRAN__ */
#ifdef __INTEL_COMPILER
  CHARACTER(LEN=*), PARAMETER :: compiler = 'Intel Fortran'
#endif /* __INTEL_COMPILER */
#ifdef __PGI
#ifdef __NVCOMPILER
  ! Note: NVHPC also defines __PGI for backwards compatibility
  CHARACTER(LEN=*), PARAMETER :: compiler = 'NVIDIA HPC Fortran'
#else
  CHARACTER(LEN=*), PARAMETER :: compiler = 'PGI Fortran'
#endif /* __NVCOMPILER */
#endif /* __PGI */
#ifdef __IBMC__
  CHARACTER(LEN=*), PARAMETER :: compiler = 'IBM XL Fortran'
#endif /* __IBMC__ */
#ifdef __FLANG
  CHARACTER(LEN=*), PARAMETER :: compiler = 'LLVM Flang'
#endif /* __FLANG */

  ! Vector dimension
  INTEGER :: N

  ! Block and grid dimensions
#ifdef _CUDA
  TYPE(dim3) :: grid, blk
#else
  INTEGER(KIND=C_INT), DIMENSION(3) :: grid, blk
#endif
  INTEGER :: Nb

  ! CUDA status/error var
#ifdef _CUDA
  INTEGER :: ierr
#else
  INTEGER(KIND=C_INT) :: ierr
#endif /* _CUDA */

#ifndef _CUDA
  ! Pointer array for CUDA kernel arguments
  TYPE(C_PTR), DIMENSION(:), ALLOCATABLE :: args
#endif /* ! _CUDA */

  ! Vectors to take the dot product with
#ifdef _CUDA
  REAL(KIND=dd), DIMENSION(:), ALLOCATABLE, DEVICE :: d_vecA, d_vecB
#else
  TYPE(C_PTR) :: d_vecA, d_vecB ! device arrays
#endif /* _CUDA */

  ! Result and check value
  REAL(KIND=dd) :: h_result, h_check ! host
#ifdef _CUDA
  REAL(KIND=dd), DEVICE :: d_result
#else
  TYPE(C_PTR) :: d_result
#endif /* _CUDA */

  ! Tolerance for error checking
  REAL(KIND=dd), PARAMETER :: tol = 10._dd**(-10)

  ! Timing subroutine
  INTRINSIC :: DATE_AND_TIME

  ! Timing vars
  INTEGER, DIMENSION(8) :: t0, t1, t2
  REAL(KIND=dd) :: time_init, time_ddot

  ! argc, argv (Fortran 2003)
  INTRINSIC :: COMMAND_ARGUMENT_COUNT, GET_COMMAND_ARGUMENT
  INTEGER :: argc
  CHARACTER(LEN=8) :: argv1, argv2 ! We only two, for N and Nb

  ! Cleanup phase
  INTEGER :: phase = 0

  ! Emit compiler identification and CUDA Fortran status
  WRITE(*,*) 'Compiled with ', compiler
#ifdef _CUDA
  WRITE(*,*) 'CUDA Fortran enabled'
#else
  WRITE(*,*) 'CUDA Fortran disabled'
#endif /* _CUDA */

  ! Check argc
  IF( COMMAND_ARGUMENT_COUNT() == 2 ) THEN

     ! Read N from argv[1]
     CALL GET_COMMAND_ARGUMENT( 1, VALUE=argv1 )
     READ( argv1, * ) N

     ! Read Nb from argv[2]
     CALL GET_COMMAND_ARGUMENT( 2, VALUE=argv2 )
     READ( argv2, * ) Nb

  ELSE

     ! Read N from standard input
     WRITE(*,*) 'Input vector length N:'
     READ(*,*) N

     ! Read Nb from standard input
     WRITE(*,*) 'Input block size Nb:'
     READ(*,*) Nb

  END IF

  ! Echo N and Nb to standard output
  WRITE(*,*) 'Using N  = ', N
  WRITE(*,*) 'Using Nb = ', Nb

#ifdef _CUDA

  ! CUDA Fortran version

  ! Allocate vectors on device (GPU) using CUDA Fortran
  ALLOCATE( d_vecA(N), STAT=ierr )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cannot allocate vector A on device, code ', ierr
     STOP
  END IF
  ALLOCATE( d_vecB(N), STAT=ierr )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cannot allocate vector B on device, code ', ierr
     DEALLOCATE( d_vecA )
     STOP
  END IF

  phase = 1

  ! Set up grid and threadblock
  grid = dim3( (N+Nb-1)/Nb, 1, 1 ) ! 1-D grid
  blk  = dim3( Nb, 1, 1 ) ! 1-D block

  ! Initialize vectors using CUDA Fortran device kernel
  CALL initVecs<<< grid, blk, 0, 0 >>>( N, d_vecA, d_vecB )

#else

  ! CUDA C++ version

  ! Allocate vectors and result variable on device (GPU) using CUDART calls
  ierr = cudaMalloc( d_vecA, N*sz_d )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cannot allocate vector A on device, code ', ierr
     STOP
  END IF
  ierr = cudaMalloc( d_vecB, N*sz_d )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cannot allocate vector B on device, code ', ierr
     ierr = cudaFree( d_vecA )
     STOP
  END IF
  ierr = cudaMalloc( d_result )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cannot allocate result variable on device, code ', ierr
     ierr = cudaFree( d_vecA )
     ierr = cudaFree( d_vecB )
     STOP
  END IF
  phase = 1

  ! Set up grid and threadblock
  grid = (/ (N+Nb-1)/Nb, 1, 1 /) ! 1-D grid
  blk  = (/ Nb, 1, 1 /) ! 1-D block

  ! Initialize vectors using CUDA C++ device kernel
  ALLOCATE( args(3) ); phase = 2
  args(1) = C_LOC( N )
  args(2) = C_LOC( d_vecA )
  args(3) = C_LOC( d_vecB )
  ierr =  cudaLaunchKernel( initVecs, grid, blk, args, 0, 0 )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cudaLaunchKernel returned ', ierr
     CALL cleanup( phase )
     STOP
  END IF
  DEALLOCATE( args ); phase = 1

#endif /* _CUDA */

  ! Synchronize device
  ierr = cudaDeviceSynchronize()
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cudaDeviceSynchronize returned ', ierr
     CALL cleanup( phase )
     STOP
  END IF

#ifdef _CUDA

  ! CUDA Fortran version

  ! Set up grid and threadblock
  grid = dim3( (N+Nb-1)/Nb, 1, 1 ) ! 1-D grid
  blk  = dim3( Nb, 1, 1 ) ! 1-D block

  ! Perform dot product using CUDA Fortran device kernel
  CALL ddot<<< grid, blk, 0, 0 >>>( N, d_vecA, d_vecB, d_result )

#else

  ! CUDA C++ version

  ! Set up grid and threadblock
  grid = (/ (N+Nb-1)/Nb, 1, 1 /) ! 1-D grid
  blk  = (/ Nb, 1, 1 /) ! 1-D block

  ! Initialize vectors using CUDA C++ device kernel
  ALLOCATE( args(4) ); phase = 2
  args(1) = C_LOC( N )
  args(2) = C_LOC( d_vecA )
  args(3) = C_LOC( d_vecB )
  args(4) = C_LOC( d_result )
  ierr =  cudaLaunchKernel( ddot, grid, blk, args, 0, 0 )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cudaLaunchKernel returned ', ierr
     CALL cleanup( phase )
     STOP
  END IF
  DEALLOCATE( args ); phase = 1

#endif /* _CUDA */

  ! Synchronize device
  ierr = cudaDeviceSynchronize()
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cudaDeviceSynchronize returned ', ierr
     CALL cleanup( phase )
     STOP
  END IF

#ifdef _CUDA

  ! CUDA Fortran version

  ! Transfer result to host using CUDA Fortran
  h_result = d_result

#else

  ! CUDA C++ version

  ! Transfer result to host using CUDART call
  ierr = cudaMemcpy( h_result, d_result, sz_z )
  IF( ierr /= 0 ) THEN
     WRITE(*,*) 'Error: cudaMemcpy returned ', ierr
     CALL cleanup( phase )
     STOP
  END IF

#endif /* _CUDA */

  ! Check value ( using relative error ) and print result to standard output
  h_check = REAL( N, KIND=dd ) * REAL( N+1, KIND=dd ) * REAL( 2*N+1, KIND=dd) &
            / 3._dd
  IF( ABS( h_result/h_check - 1 ) > tol ) THEN
     WRITE(*,*) 'Error! Result = ', h_result, ' when it should be ', h_check
  ELSE
     WRITE(*,*) 'Success! Result = ', h_result
  END IF

  ! Clean up
  CALL cleanup( phase )

  STOP

CONTAINS

  SUBROUTINE cleanup( phase )
    IMPLICIT NONE

    ! Input argument
    INTEGER, INTENT(IN) :: phase

#ifndef _CUDA
    IF( phase >= 2 ) THEN
       DEALLOCATE( args )
    END IF
#endif /* ! _CUDA */

    IF( phase >= 1 ) THEN
#ifdef _CUDA
       DEALLOCATE( d_vecA )
       DEALLOCATE( d_vecB )
#else
       ierr = cudaFree( d_vecA )
       ierr = cudaFree( d_vecB )
       ierr = cudaFree( d_result )
#endif /* _CUDA */
    END IF

    RETURN
  END SUBROUTINE cleanup

END PROGRAM cuf_dadd
