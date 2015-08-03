
INC_DIR = -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5\common\inc"
LIB_DIR = -L/cygdrive/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v6.5/lib/Win32
LIB_FILE = -lcuda

.PHONY: all clean

all: clean main.exe

clean:
	rm -rf *.exe *.o *.out *.obj

%.exe: %.obj
	nvcc $(LIB_DIR) $(LIB_FILE) $^ -o $@

%.obj: %.cu
	nvcc -c $(INC_DIR) $^ -o $@

