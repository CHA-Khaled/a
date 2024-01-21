CC=gcc

CFLAGS=-march=native -g3

OFLAGS=-O0 -fopenmp -fopt-info-all=nbody.gcc.optrpt

all: nbody3D

nbody3D: nbody.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@ -lm

clean:
	rm -Rf *~ nbody3D *.optrpt

