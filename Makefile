############################################
# Treba makefile                           #
# To compile with NVIDIA CUDA support run: #
# make CUDA=1                              #
############################################

PREFIX = /usr/local
CC = gcc -Wall
BINPREFIX = $(PREFIX)/bin/
MANPREFIX = $(PREFIX)/share/man/man1/
RM = /bin/rm -f
# NOTE: `-fcommon` keeps legacy global definitions linkable on modern GCC
# (newer GCC defaults to `-fno-common`, which breaks this codebase).
CFLAGS = -O3 -ffast-math -fcommon
HOST_ARCH := $(shell uname -m)

ifeq ($(CUDA),1)
	CUDA_INSTALL_PATH ?= /usr/local/cuda
	CUDA_ARCH_FLAGS ?= -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
	NVCCFLAGS = -m64 -I$(CUDA_INSTALL_PATH)/include $(CUDA_ARCH_FLAGS)
ifeq ($(HOST_ARCH),aarch64)
	# CUDA 12 + recent glibc on aarch64 may fail parsing math-vector typedefs.
	NVCCFLAGS += -D__GNUC__=8 -D__GNUC_MINOR__=0
endif
	CFLAGS += -DUSE_CUDA
	LFLAGS = -lm -lpthread -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_INSTALL_PATH)/lib -lcudart -lgsl -lgslcblas
	TREBADEPS = treba.o dffa.o gibbs.o observations.o io.o treba.h treba_cuda.o fastlogexp.h
	TREBACMD = $(CC) $(CFLAGS) -DUSE_CUDA -o treba treba_cuda.o treba.o dffa.o gibbs.o observations.o io.o $(LFLAGS)
else
	LFLAGS = -lm -lpthread -lgsl -lgslcblas
	TREBADEPS = treba.o dffa.o gibbs.o observations.o io.o treba.h fastlogexp.h
	TREBACMD = $(CC) $(CFLAGS) -o treba treba.o dffa.o gibbs.o observations.o io.o $(LFLAGS)
endif


all:	treba

treba:	$(TREBADEPS)
	$(TREBACMD)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

treba_cuda.o: treba_cuda.cu treba.h fastlogexp.h
	nvcc $(NVCCFLAGS) -o treba_cuda.o -c treba_cuda.cu

clean:
	$(RM) treba treba.o dffa.o gibbs.o observations.o io.o treba_cuda.o

install: treba treba.1
	-@if [ ! -d $(BINPREFIX) ]; then mkdir -p $(BINPREFIX); fi
	-@if [ ! -d $(MANPREFIX) ]; then mkdir -p $(MANPREFIX); fi
	cp treba $(BINPREFIX)
	cp ./man/treba.1 $(MANPREFIX)
