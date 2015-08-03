
INC_DIR = -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5\common\inc"
LIB_DIR = -L/cygdrive/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v6.5/lib/Win32
LIB_DIR += -L./
LIB_FILE = -lcuda

PROJECT = matrixMul_main
SRCS = $(PROJECT).cpp matrixMul_kernel.cpp
OBJS = $(SRCS:.cpp=.obj)

EXE = $(PROJECT).exe

.PHONY: all clean

all: clean matrixMul_main.exe

clean:
	rm -rf *.exe *.o *.out *.obj *.exp *.lib

$(EXE): $(OBJS)
	nvcc $(LIB_DIR) $(LIB_FILE) $^ -o $@

%.lib: %.obj
	mv $^ $@

%.obj: %.cpp
	nvcc -c $(INC_DIR) $^ -o $@

%.obj: %.cu
	nvcc -c $(INC_DIR) $^ -o $@

