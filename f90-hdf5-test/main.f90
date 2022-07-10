PROGRAM test_hdf5

  USE mod_prec
  USE mod_hdf5

  IMPLICIT NONE

  ! Matrix array dimensions
  INTEGER, PARAMETER :: N = 1024
  INTEGER, PARAMETER :: Nmat = 16

  ! Matrix array
  REAL(KIND=dd), DIMENSION(:,:,:), ALLOCATABLE :: matA, matB

  ! HDF5 filename
  CHARACTER(LEN=16) :: fname = 'test.hdf5'

  ! Loop indices
  INTEGER :: i, j, k

  ! Initialize HDF5
  CALL hdf5_init()
  WRITE(*,*) 'Initialized HDF5'

  ! Allocate and initialize matrix array
  ALLOCATE( matA( N, N, Nmat ) )
  DO k = 1, Nmat
     DO j = 1, N
        DO i = 1, N
           matA(i,j,k) = REAL( i+j, KIND=dd ) + 1.E-2_dd * REAL( k, KIND=dd )
        END DO ! i
     END DO ! j
  END DO ! k

  ! Create/overwrite HDF5 file
  CALL hdf5_newfile( TRIM(fname) )

  ! Write metadata to HDF5 file
  WRITE(*,*) 'Writing parameters to HDF5 file ', TRIM(fname)
  CALL hdf5_newgroup( TRIM(fname), '/', 'parameters' )
  CALL hdf5_write( TRIM(fname), '/parameters/', 'N', N )
  WRITE(*,*) 'Wrote N = ', N
  CALL hdf5_write( TRIM(fname), '/parameters/', 'Nmat', Nmat )
  WRITE(*,*) 'Wrote Nmat = ', Nmat

  ! Write matrix array using HDF5
  WRITE(*,*) 'Writing matrix array to HDF5 file ', TRIM(fname)
  CALL hdf5_write( TRIM(fname), '/', 'mat', matA(1,1,1), (/ N, N, Nmat /) )

  ! Clean up matrix array
  DEALLOCATE( matA )

  ! Read matrix dimension from HDF5 file
  WRITE(*,*) 'Reading parameters from HDF5 file ', TRIM(fname)
  CALL hdf5_read( TRIM(fname), '/parameters/', 'N', i )
  WRITE(*,*) 'Read N = ', i
  CALL hdf5_read( TRIM(fname), '/parameters/', 'Nmat', k )
  WRITE(*,*) 'Read Nmat = ', k

  ! Allocate matrix and read from HDF5 file
  ALLOCATE( matB(i,i,k) )
  WRITE(*,*) 'Reading matrix array from HDF5 file ', TRIM(fname)
  CALL hdf5_read( TRIM(fname), '/', 'mat', matB(1,1,1), (/ i, i, k /) )

  ! Print last element of matrix
  WRITE(*,*) 'mat(', i, ', ', i, ', ', k, ') = ', matB(i,i,k)

  ! Clean up
  DEALLOCATE( matB )

  STOP
END PROGRAM test_hdf5
