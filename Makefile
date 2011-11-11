CXX=c++
CFLAGS=-Wall -I../Torch3/core -I../Torch3/kernels
LIBS=-L../Torch3/lib/Darwin_OPT_FLOAT -l torch

multisvm: multisvm.cpp
	$(CXX) $(CFLAGS) $(LIBS) -o $@ $<
