cmake_minimum_required(VERSION 3.16)
project(KokkosDotProduct)
include(cmake/FindKokkos.cmake)

# Detect which Kokkos backend is in use
foreach(DEV ${Kokkos_DEVICES})
  if(${Kokkos_ENABLE_${DEV}})
    string(TOLOWER ${DEV} BACKEND)
    break()
  endif()
endforeach()

add_executable(ddot.${BACKEND}.x ddot.kokkos.cpp)
target_link_libraries(ddot.${BACKEND}.x Kokkos::kokkos)
