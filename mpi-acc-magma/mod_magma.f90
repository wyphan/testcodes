MODULE mod_magma

  USE ISO_C_BINDING
  IMPLICIT NONE

!===============================================================================
! Interfaces to MAGMA needed for the MPI + MAGMA test code
! Those that are already available are copied from MAGMA 2.5.4 source tarball
! - fortran/magma2.F90
! - fortran/magma2_zfortran.F90
!===============================================================================

!! =============================================================================
!! Parameter constants from magma_types.h
integer(c_int), parameter ::   &
    MagmaNoTrans       = 111,  &
    MagmaTrans         = 112,  &
    MagmaConjTrans     = 113

!! =============================================================================
!! Fortran interfaces to C functions
interface

    !! -------------------------------------------------------------------------
    !! initialize
    subroutine magma_init() &
    bind(C, name="magma_init")
        use iso_c_binding
    end subroutine

    subroutine magma_finalize() &
    bind(C, name="magma_finalize")
        use iso_c_binding
    end subroutine

    !! -------------------------------------------------------------------------
    !! device support
    subroutine magma_set_device( dev ) &
    bind(C, name="magma_setdevice")
        use iso_c_binding
        integer(c_int), value :: dev
    end subroutine

    !! -------------------------------------------------------------------------
    !! queue support
    subroutine magma_queue_create_internal( dev, queue_ptr, func, file, line ) &
    bind(C, name="magma_queue_create_internal")
        use iso_c_binding
        integer(c_int), value :: dev
        type(c_ptr), target :: queue_ptr  !! queue_t*
        character(c_char) :: func, file
        integer(c_int), value :: line
    end subroutine

    subroutine magma_queue_destroy_internal( queue, func, file, line ) &
    bind(C, name="magma_queue_destroy_internal")
        use iso_c_binding
        type(c_ptr), value :: queue  !! queue_t
        character(c_char) :: func, file
        integer(c_int), value :: line
    end subroutine

    subroutine magma_queue_sync_internal( queue, func, file, line ) &
    bind(C, name="magma_queue_sync_internal")
        use iso_c_binding
        type(c_ptr), value :: queue  !! queue_t
        character(c_char) :: func, file
        integer(c_int), value :: line
    end subroutine

    !! -------------------------------------------------------------------------
    !! BLAS (matrices in GPU memory)
    subroutine magmablas_dgemv( &
        trans, m, n, &
        alpha, dA, ldda, &
               dx, incx, &
        beta,  dy, incy, &
        queue ) &
    bind(C, name="magmablas_zgemv")
        use iso_c_binding
        integer(c_int), value :: trans, m, n, ldda, incx, incy
        real(c_double), value :: alpha, beta
        type(c_ptr),    value :: dA, dx, dy
        type(c_ptr),    value :: queue  !! queue_t
    end subroutine

    
!! =============================================================================
!! Fortran routines & functions
contains

    !! -------------------------------------------------------------------------
    !! queue support
    subroutine magma_queue_create( dev, queue_ptr )
        use iso_c_binding
        integer(c_int), value :: dev
        type(c_ptr), target :: queue_ptr  !! queue_t*
        
        call magma_queue_create_internal( &
                dev, queue_ptr, &
                "magma_queue_create" // c_null_char, &
                __FILE__ // c_null_char, &
                __LINE__ )
    end subroutine

    subroutine magma_queue_destroy( queue )
        use iso_c_binding
        type(c_ptr), value :: queue  !! queue_t
        
        call magma_queue_destroy_internal( &
                queue, &
                "magma_queue_destroy" // c_null_char, &
                __FILE__ // c_null_char, &
                __LINE__ )
    end subroutine

    subroutine magma_queue_sync( queue )
        use iso_c_binding
        type(c_ptr), value :: queue  !! queue_t
        
        call magma_queue_sync_internal( &
                queue, &
                "magma_queue_sync" // c_null_char, &
                __FILE__ // c_null_char, &
                __LINE__ )
    end subroutine

end module
