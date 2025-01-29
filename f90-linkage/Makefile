FC = gfortran
FCFLAGS = -O2 -ggdb
CC = gcc
CCFLAGS = -O2 -ggdb
LDFLAGS = -g -lc

EXE = linkage.$(FC).$(CC).x

.PHONY: all

${EXE}: main.o my_module.o my_explicit_iface_fn.o my_external_f77_sub.o my_c_fn.o
	$(FC) -o $(EXE) $^ $(LDFLAGS)

%.f:
%.o: %.f
	$(FC) $(FCFLAGS) -c -o $@ $<

%.f90:
%.o: %.f90
	$(FC) $(FCFLAGS) -c -o $@ $<

%.c:
%.o: %.c
	${CC} ${CCFLAGS} -c -o $@ $<

clean:
	-rm -rf *.o *.mod *.smod

distclean: clean
	-rm -rf $(EXE)
