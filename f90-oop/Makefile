FC = gfortran
FCFLAGS = -O2 -ggdb
LDFLAGS = -ggdb

EXE = clasmech.$(FC).x

.PHONY: all

all: $(EXE)

$(EXE): main.o vec_m.o body_m.o body_construct_s.o body_motion_s.o
	$(FC) -o $(EXE) $^ $(LDFLAGS)

main.o: vec_m.o body_m.o body_construct_s.o body_motion_s.o
body_construct_s.o: body_m.o
body_motion_s.o: body_m.o
body_m.o: vec_m.o

%.f90:
%.o: %.f90
	$(FC) $(FCFLAGS) -c -o $@ $<

clean:
	-rm -rf *.o *.mod *.smod

distclean: clean
	-rm -rf $(EXE)
