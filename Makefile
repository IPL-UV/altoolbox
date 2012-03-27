TORCH=../Torch3
CXX=c++
CXXFLAGS=-Wall -I$(TORCH)/core -I$(TORCH)/kernels -g

ifeq ($(shell uname),Linux)
	LIBS=-L$(TORCH)/lib/Linux_OPT_FLOAT
else
	#LIBS=-L$(TORCH)/lib/Darwin_OPT_FLOAT
	#CXX=i586-mingw32msvc-g++
	#LIBS=-L$(TORCH)/lib/Windows_OPT_FLOAT
	CXX=x86_64-w64-mingw32-g++
	LIBS=-L$(TORCH)/lib/Windows_64_OPT_FLOAT
endif

multisvm: multisvm.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $<  -l torch

multisvm.exe: multisvm.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $<  -l torch

