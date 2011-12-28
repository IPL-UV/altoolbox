TORCH=../Torch3
CXX=c++
CXXFLAGS=-Wall -I$(TORCH)/core -I$(TORCH)/kernels -g

ifeq ($(shell uname),Linux)
	LIBS=-L$(TORCH)/lib/Linux_OPT_FLOAT
else
	LIBS=-L$(TORCH)/lib/Darwin_OPT_FLOAT
endif

multisvm: multisvm.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $<  -l torch

