NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -w -O3 -arch=sm_61 -DCRAM_DEBUG -I./src/include -I./src/kernels

MPI_INC = /usr/local/openmpi/include
MPI_LIB = /usr/local/openmpi/lib

ifeq ($(OS),Windows_NT)
    OS_DETECTED := Windows
    PATH_SEP := /
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_DETECTED := Linux
        NVCCFLAGS += -Xcompiler -fPIC -I$(MPI_INC) -L$(MPI_LIB) -lmpi -lcublas
        PATH_SEP := /
    endif
endif

OBJ_PATH = .$(PATH_SEP)objectfiles

TARGET = ldm

SRCS = src/main.cu
OBJS = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS:.cu=.o))

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

$(OBJ_PATH)$(PATH_SEP)%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
ifeq ($(OS_DETECTED),Windows)
	del /Q /F $(subst /,\,$(OBJS)) $(TARGET).exe $(TARGET).exp $(TARGET).lib
else
	rm -f $(OBJS) $(TARGET)
endif

$(OBJS): | $(OBJ_PATH)

$(OBJ_PATH):
	mkdir -p $(OBJ_PATH)
