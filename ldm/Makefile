NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -w -O3 -arch=sm_80 -DLDM_DEBUG_ENS -std=c++17 -I./src/include -I./src/kernels

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

TARGET = ldm_main
TARGET_REFACTORED = ldm_refactored

SRCS = src/main.cu src/ldm_eki.cu src/ldm_eki_logger.cu src/ldm_ensemble_init.cu src/kernels/ldm_ensemble_kernels.cu
SRCS_REFACTORED = src/main_refactored.cu src/ldm_integration.cu src/ldm_eki.cu src/ldm_eki_logger.cu src/ldm_ensemble_init.cu src/kernels/ldm_ensemble_kernels.cu
OBJS = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS:.cu=.o))
OBJS_REFACTORED = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS_REFACTORED:.cu=.o))

all: $(TARGET) $(TARGET_REFACTORED)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

$(TARGET_REFACTORED): $(OBJS_REFACTORED)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS_REFACTORED)

$(OBJ_PATH)$(PATH_SEP)%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
ifeq ($(OS_DETECTED),Windows)
	del /Q /F $(subst /,\,$(OBJS)) $(subst /,\,$(OBJS_REFACTORED)) $(TARGET).exe $(TARGET_REFACTORED).exe $(TARGET).exp $(TARGET).lib $(TARGET_REFACTORED).exp $(TARGET_REFACTORED).lib
else
	rm -f $(OBJS) $(OBJS_REFACTORED) $(TARGET) $(TARGET_REFACTORED)
endif

$(OBJS) $(OBJS_REFACTORED): | $(OBJ_PATH)

$(OBJ_PATH):
	mkdir -p $(OBJ_PATH)
	mkdir -p $(OBJ_PATH)/src
	mkdir -p $(OBJ_PATH)/src/kernels
