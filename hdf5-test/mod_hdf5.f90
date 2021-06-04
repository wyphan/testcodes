MODULE mod_hdf5

  USE mod_prec
  USE hdf5

  IMPLICIT NONE

  PRIVATE
  PUBLIC :: hdf5_init, hdf5_fin
  PUBLIC :: hdf5_newfile, hdf5_newgroup, hdf5_read, hdf5_write

  INTERFACE hdf5_read
     MODULE PROCEDURE hdf5_read_i4, hdf5_read_d, hdf5_read_array_d
  END INTERFACE hdf5_read

  INTERFACE hdf5_write
     MODULE PROCEDURE hdf5_write_i4, hdf5_write_d, hdf5_write_array_d
  END INTERFACE hdf5_write

CONTAINS

!===============================================================================
! Initializes HDF5 library

  SUBROUTINE hdf5_init()
    IMPLICIT NONE

    ! Internal variables
    INTEGER :: ierr

    CALL h5open_f(ierr)
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_init: h5open_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_init

!===============================================================================
! Finalizes HDF5 library

  SUBROUTINE hdf5_fin()
    IMPLICIT NONE

    ! Internal variables
    INTEGER :: ierr

    CALL h5close_f(ierr)
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_fin: h5close_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_fin

!===============================================================================
! Creates a new, empty HDF5 file

  SUBROUTINE hdf5_newfile( fname )
    IMPLICIT NONE

    ! Input argument
    CHARACTER(LEN=*), INTENT(IN) :: fname

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root
    INTEGER :: ierr
    LOGICAL :: exist

    ! First, check whether HDF5 file already exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( exist ) THEN
       WRITE(*,*) 'hdf5_newfile: Warning: overwriting ', TRIM(fname)
    ELSE
       WRITE(*,*) 'hdf5_newfile: Creating new file ', TRIM(fname)
    END IF

    ! Now, actually create/overwrite the HDF5 file
    CALL h5fcreate_f( TRIM(fname), H5F_ACC_TRUNC_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newfile: h5fcreate_f error ', ierr       
    END IF

    ! Close the HDF5 file
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newfile: h5fclose_f error ', ierr       
    END IF

    RETURN
  END SUBROUTINE hdf5_newfile

!===============================================================================
! Creates a new, empty group in an existing HDF5 file with the default
! property lists (LCPL, GCPL, GAPL)

  SUBROUTINE hdf5_newgroup( fname, path, group )
    IMPLICIT NONE

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, group

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5group
    INTEGER :: ierr
    LOGICAL :: exist

    ! First, check whether HDF5 file already exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_newgroup: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read+write mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDWR_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newgroup: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as an existing group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newgroup: Error: Cannot open path in file ', TRIM(path)
       WRITE(*,*) 'hdf5_newgroup: h5gopen_f error ', ierr
       STOP
    END IF

    ! Create the new group
    CALL h5gcreate_f( h5path, TRIM(group), h5group, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newgroup: Error: Cannot create new group ', TRIM(group)
       WRITE(*,*) 'hdf5_newgroup: h5gcreate_f error ', ierr
       STOP
    END IF

    ! Close both groups
    CALL h5gclose_f( h5group, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newgroup: first h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newgroup: second h5gclose_f error ', ierr
       STOP
    END IF

    ! Close HDF5 file
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_newgroup: h5fclose_f error ', ierr       
    END IF

    RETURN
  END SUBROUTINE hdf5_newgroup

!===============================================================================
! Reads a scalar integer value from HDF5 file at the selected path using
! default property lists (DCPL, LCPL, DAPL, DXPL)

  SUBROUTINE hdf5_read_i4( fname, path, name, val )
    IMPLICIT NONE

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, name
    INTEGER, INTENT(OUT) :: val

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5dset
    INTEGER(KIND=HSIZE_T), DIMENSION(1) :: h5dims
    INTEGER :: ierr
    LOGICAL :: exist

    ! First, check whether HDF5 file exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_read_i4: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read-only mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDONLY_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as a group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: Error: Cannot open path in file ', TRIM(path)
       WRITE(*,*) 'hdf5_read_i4: h5gopen_f error ', ierr
       STOP
    END IF

    ! Open the dataset using default property lists (DCPL, LCPL, DAPL)
    CALL h5dopen_f( h5path, TRIM(name), h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: Error: Cannot open dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_read_i4: h5dopen_f error ', ierr
       STOP
    END IF

    ! Read data from dataset using default data transfer property list (DXPL)
    h5dims(1) = 1
    CALL h5dread_f( h5dset, H5T_NATIVE_INTEGER, val, h5dims, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: Error: Cannot read dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_read_i4: h5dread_f error ', ierr
       STOP
    END IF

    ! Clean up and close HDF5 file
    CALL h5dclose_f( h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: h5dclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_i4: h5fclose_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_read_i4

!===============================================================================
! Writes a scalar integer value into HDF5 file at the selected path using
! default property lists (DCPL, LCPL, DAPL, DXPL)

  SUBROUTINE hdf5_write_i4( fname, path, name, val )
    IMPLICIT NONE

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, name
    INTEGER, INTENT(IN) :: val

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5space, h5dset
    INTEGER(KIND=HSIZE_T), DIMENSION(1) :: h5dims
    INTEGER :: ierr
    LOGICAL :: exist

    ! First, check whether HDF5 file exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_write_i4: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read+write mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDWR_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as a group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: Error: Cannot open path in file ', TRIM(path)
       WRITE(*,*) 'hdf5_write_i4: h5gopen_f error ', ierr
       STOP
    END IF

    ! Create a scalar dataspace to store the data
    CALL h5screate_f( H5S_SCALAR_F, h5space, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: Error: Cannot create new scalar dataspace'
       WRITE(*,*) 'hdf5_write_i4: h5screate_f error ', ierr
       STOP
    END IF
    
    ! Create the dataset associated with the dataspace using default property
    ! lists (DCPL, LCPL, DAPL)
    CALL h5dcreate_f( h5path, TRIM(name), H5T_NATIVE_DOUBLE, h5space, h5dset, &
                      ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: Error: Cannot create new dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_write_i4: h5dcreate_f error ', ierr
       STOP
    END IF

    ! Write data into dataset using default data transfer property list (DXPL)
    h5dims(1) = 1
    CALL h5dwrite_f( h5dset, H5T_NATIVE_INTEGER, val, h5dims, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: Error: Cannot write dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_write_i4: h5dwrite_f error ', ierr
       STOP
    END IF

    ! Clean up and close HDF5 file
    CALL h5sclose_f( h5space, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: h5sclose_f error ', ierr
       STOP
    END IF
    CALL h5dclose_f( h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: h5dclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_i4: h5fclose_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_write_i4

!===============================================================================
! Reads a scalar double precision value from HDF5 file at the selected path
! using default property lists (DCPL, LCPL, DAPL, DXPL)

  SUBROUTINE hdf5_read_d( fname, path, name, val )
    IMPLICIT NONE

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, name
    REAL(KIND=dd), INTENT(OUT) :: val

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5dset
    INTEGER(KIND=HSIZE_T), DIMENSION(1) :: h5dims
    INTEGER :: ierr
    LOGICAL :: exist

    ! First, check whether HDF5 file exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_read_d: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read-only mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDONLY_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as a group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: Error: Cannot open path in file ', TRIM(path)
       WRITE(*,*) 'hdf5_read_d: h5gopen_f error ', ierr
       STOP
    END IF

    ! Open the dataset using default property lists (DCPL, LCPL, DAPL)
    CALL h5dopen_f( h5path, TRIM(name), h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: Error: Cannot open dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_read_d: h5dopen_f error ', ierr
       STOP
    END IF

    ! Read data from dataset using default data transfer property list (DXPL)
    h5dims(1) = 1
    CALL h5dread_f( h5dset, H5T_NATIVE_DOUBLE, val, h5dims, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: Error: Cannot read dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_read_d: h5dread_f error ', ierr
       STOP
    END IF

    ! Clean up and close HDF5 file
    CALL h5dclose_f( h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: h5dclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_d: h5fclose_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_read_d

!===============================================================================
! Writes a scalar double precision value into HDF5 file at the selected path
! using default property lists (DCPL, LCPL, DAPL, DXPL)

  SUBROUTINE hdf5_write_d( fname, path, name, val )
    IMPLICIT NONE

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, name
    REAL(KIND=dd), INTENT(IN) :: val

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5space, h5dset
    INTEGER(KIND=HSIZE_T), DIMENSION(1) :: h5dims
    INTEGER :: ierr
    LOGICAL :: exist

    ! First, check whether HDF5 file exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_write_d: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read+write mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDWR_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as a group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: Error: Cannot open path in file ', TRIM(path)
       WRITE(*,*) 'hdf5_write_d: h5gopen_f error ', ierr
       STOP
    END IF

    ! Create a scalar dataspace to store the data
    CALL h5screate_f( H5S_SCALAR_F, h5space, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: Error: Cannot create new scalar dataspace'
       WRITE(*,*) 'hdf5_write_d: h5screate_f error ', ierr
       STOP
    END IF
    
    ! Create the dataset associated with the dataspace using default property
    ! lists (DCPL, LCPL, DAPL)
    CALL h5dcreate_f( h5path, TRIM(name), H5T_NATIVE_DOUBLE, h5space, h5dset, &
                      ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: Error: Cannot create new dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_write_d: h5dcreate_f error ', ierr
       STOP
    END IF

    ! Write data into dataset using default data transfer property list (DXPL)
    h5dims(1) = 1
    CALL h5dwrite_f( h5dset, H5T_NATIVE_DOUBLE, val, h5dims, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: Error: Cannot write dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_write_d: h5dwrite_f error ', ierr
       STOP
    END IF

    ! Clean up and close HDF5 file
    CALL h5sclose_f( h5space, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: h5sclose_f error ', ierr
       STOP
    END IF
    CALL h5dclose_f( h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: h5dclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_d: h5fclose_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_write_d

!===============================================================================
! Reads a double precision array from HDF5 file at the selected path using
! default property lists (DCPL, LCPL, DAPL, DXPL)
! Note: Please pass only an element of the val array,
!       e.g., val(1,1,1) instead of val(:,:,1) or just val
  
  SUBROUTINE hdf5_read_array_d( fname, path, name, val, dims )
    USE ISO_C_BINDING, ONLY: C_LOC, C_F_POINTER

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, name
    REAL(KIND=dd), TARGET, INTENT(OUT) :: val
    INTEGER, DIMENSION(:), INTENT(IN) :: dims

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5dset
    INTEGER(KIND=HSIZE_T), DIMENSION(SIZE(dims)) :: h5dims
    REAL(KIND=dd), DIMENSION(:), POINTER :: buf
    INTEGER(KIND=dl) :: sz_buf
    INTEGER :: ierr, dim
    LOGICAL :: exist

    ! First, check whether HDF5 file exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_read_array_d: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read-only mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDONLY_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as a group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: Error: Cannot open path in file ', &
                  TRIM(path)
       WRITE(*,*) 'hdf5_read_array_d: h5gopen_f error ', ierr
       STOP
    END IF

    ! Open the dataset using default property lists (DCPL, LCPL, DAPL)
    CALL h5dopen_f( h5path, TRIM(name), h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: Error: Cannot open dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_read_array_d: h5dopen_f error ', ierr
       STOP
    END IF

    ! Convert dims to HSIZE_T
    h5dims(:) = dims(:)

    ! C pointer trickery: cast double -> void* -> double*
    ! Note: no need to ALLOCATE buf, see discussion at https://stackoverflow.com/questions/67843153/fortran-nullify-vs-deallocate
    sz_buf = PRODUCT(dims)
    CALL C_F_POINTER( C_LOC(val), buf, (/sz_buf/) )

    ! Read data from dataset through buffer
    CALL h5dread_f( h5dset, H5T_NATIVE_DOUBLE, buf, h5dims, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: Error: Cannot read dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_read_array_d: h5dread_f error ', ierr
       NULLIFY(buf)
       STOP
    END IF

    ! Clean up and close HDF5 file
    NULLIFY(buf)
    CALL h5dclose_f( h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: h5dclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_read_array_d: h5fclose_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_read_array_d

!===============================================================================
! Writes a double precision array into HDF5 file at the selected path using
! default property lists (DCPL, LCPL, DAPL, DXPL)
! Note: Please pass only an element of the val array,
!       e.g., val(1,1,1) instead of val(:,:,1) or just val

  SUBROUTINE hdf5_write_array_d( fname, path, name, val, dims )
    USE ISO_C_BINDING, ONLY: C_LOC, C_F_POINTER

    ! Input arguments
    CHARACTER(LEN=*), INTENT(IN) :: fname, path, name
    REAL(KIND=dd), TARGET, INTENT(IN) :: val
    INTEGER, DIMENSION(:), INTENT(IN) :: dims

    ! Internal variables
    INTEGER(KIND=HID_T) :: h5root, h5path, h5space, h5dset
    INTEGER(KIND=HSIZE_T), DIMENSION(SIZE(dims)) :: h5dims
    REAL(KIND=dd), DIMENSION(:), POINTER :: buf
    INTEGER(KIND=dl) :: sz_buf
    INTEGER :: ierr, dim
    LOGICAL :: exist

    ! First, check whether HDF5 file exists
    INQUIRE( FILE=TRIM(fname), EXIST=exist )
    IF( .NOT. exist ) THEN
       WRITE(*,*) 'hdf5_write_array_d: Error: file not found ', TRIM(fname)
       STOP
    END IF

    ! Open the file in read+write mode
    CALL h5fopen_f( TRIM(fname), H5F_ACC_RDWR_F, h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: h5fopen_f error ', ierr
       STOP
    END IF

    ! Open the pre-existing path in the file as a group
    CALL h5gopen_f( h5root, TRIM(path), h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: Error: Cannot open path in file ', &
                  TRIM(path)
       WRITE(*,*) 'hdf5_write_array_d: h5gopen_f error ', ierr
       STOP
    END IF

    ! Convert dims to HSIZE_T
    h5dims(:) = dims(:)

    ! Create a simple dataspace to store the data
    CALL h5screate_simple_f( SIZE(dims), h5dims, h5space, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: Error: Cannot create new simple dataspace'
       WRITE(*,*) 'hdf5_write_array_d: h5screate_simple_f error ', ierr
       STOP
    END IF

    ! Create the dataset associated with the dataspace using default property
    ! lists (DCPL, LCPL, DAPL)
    CALL h5dcreate_f( h5path, TRIM(name), H5T_NATIVE_DOUBLE, h5space, h5dset, &
                      ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: Error: Cannot create new dataset ', &
                  TRIM(name)
       WRITE(*,*) 'hdf5_write_array_d: h5dcreate_f error ', ierr
       STOP
    END IF

    ! C pointer trickery: cast double -> void* -> double*
    ! Note: no need to ALLOCATE buf, see discussion at https://stackoverflow.com/questions/67843153/fortran-nullify-vs-deallocate
    sz_buf = PRODUCT(dims)
    CALL C_F_POINTER( C_LOC(val), buf, (/sz_buf/) )

    ! Write data into dataset through buffer using default data transfer
    ! property list (DXPL)
    CALL h5dwrite_f( h5dset, H5T_NATIVE_DOUBLE, buf, h5dims, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: Error: Cannot write dataset ', TRIM(name)
       WRITE(*,*) 'hdf5_write_array_d: h5dwrite_f error ', ierr
       NULLIFY(buf)
       STOP
    END IF

    ! Clean up and close HDF5 file
    NULLIFY(buf)
    CALL h5sclose_f( h5space, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: h5sclose_f error ', ierr
       STOP
    END IF
    CALL h5dclose_f( h5dset, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: h5dclose_f error ', ierr
       STOP
    END IF
    CALL h5gclose_f( h5path, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: h5gclose_f error ', ierr
       STOP
    END IF
    CALL h5fclose_f( h5root, ierr )
    IF( ierr /= 0 ) THEN
       WRITE(*,*) 'hdf5_write_array_d: h5fclose_f error ', ierr
       STOP
    END IF

    RETURN
  END SUBROUTINE hdf5_write_array_d

!===============================================================================

END MODULE mod_hdf5
