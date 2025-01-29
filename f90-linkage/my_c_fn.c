#include "stdio.h"

int my_c_fn (int arg) {
  printf("my_c_fn is a C function called through an ISO_C_BINDING interface with argument %d\n", arg);
  return 0;
}
