MODULE mod_prec

  IMPLICIT NONE

  ! Portable precision constants
  INTEGER, PARAMETER :: dl = SELECTED_INT_KIND(18)       ! INT64
  INTEGER, parameter :: ds = SELECTED_REAL_KIND(6,30)    ! FP32
  INTEGER, parameter :: dd = SELECTED_REAL_KIND(14,100)  ! FP64
  INTEGER, PARAMETER :: dc = KIND( (0.0_ds, 1.0_ds) )
  INTEGER, PARAMETER :: dz = KIND( (0.0_dd, 1.0_dd) )

END MODULE mod_prec
