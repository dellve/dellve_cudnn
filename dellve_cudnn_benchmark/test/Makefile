NVCC=nvcc
ARCH?=sm_30

BIN_DIR?=bin
SRC_DIR?=src
INC_DIR=../include

CUDA_PATH?=/usr/local/cuda
CUDA_INCLUDE=$(CUDA_PATH)/include
CUDA_LIB64=$(CUDA_PATH)/lib64

CUDNN_PATH?=/usr/local/cudnn
CUDNN_INCLUDE=$(CUDNN_PATH)/include
CUDNN_LIB64=$(CUDNN_PATH)/lib64

CC=$(CUDA_PATH)/bin/$/$(NVCC)
IFLAGS=-I $(INC_DIR) -I $(CUDA_INCLUDE) -I $(CUDNN_INCLUDE)
LFLAGS=-L $(CUDA_LIB64) -L $(CUDNN_LIB64) -lcudnn -lcurand -lpthread
NVCC_FLAGS=$(IFLAGS) $(LFLAGS) -arch=$(ARCH) -std=c++11

MKDIR=mkdir -p
RM_RF=rm -rf

##############################
# Add programs here
##############################
TEST=basic_curand
##############################

TEST_OUT=$(addprefix $(BIN_DIR)/, $(TEST:=.test))

all: bin $(TEST_OUT) 

bin:
	$(MKDIR) $(BIN_DIR)

$(BIN_DIR)/%.test: $(SRC_DIR)/%.cu $(HEADERS)
	$(CC) $< -o $@ $(NVCC_FLAGS)

clean:
	$(RM_RF) $(BIN_DIR)

rebuild: clean all
