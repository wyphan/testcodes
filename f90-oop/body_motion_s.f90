SUBMODULE(body_m) body_motion_s
  IMPLICIT NONE

CONTAINS

  MODULE PROCEDURE body_position_at_time
    pos = body%r + time * body%v + 0.5_dp * body%a * time**2
  END PROCEDURE body_position_at_time
    
  MODULE PROCEDURE body_velocity_at_time
    vel = body%v + time * body%a
  END PROCEDURE body_velocity_at_time

END SUBMODULE body_motion_s