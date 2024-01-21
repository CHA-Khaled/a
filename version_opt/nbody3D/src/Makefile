CC=clang

CFLAGS=-march=native -g3

OFLAGS=-Ofast -fopenmp -Rpass=loop-vectorize

all: nbody3D

nbody3D: nbody.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@ -lm

clean:
	rm -Rf *~ nbody3D *.optrpt

