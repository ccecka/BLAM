#/******************************************************************************
# * Copyright (C) 2016-2017, Cris Cecka.  All rights reserved.
# * Copyright (C) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# *
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *     * Redistributions of source code must retain the above copyright
# *       notice, this list of conditions and the following disclaimer.
# *     * Redistributions in binary form must reproduce the above copyright
# *       notice, this list of conditions and the following disclaimer in the
# *       documentation and/or other materials provided with the distribution.
# *     * Neither the name of the NVIDIA CORPORATION nor the
# *       names of its contributors may be used to endorse or promote products
# *       derived from this software without specific prior written permission.
# *
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************/

####################
## Makefile Setup ##
####################

# Get the host-name if empty
ifeq ($(host-name),)
	host-name := $(shell hostname)
endif
# Get the kernel-name if empty
ifeq ($(kernel-name),)
	kernel-name := $(shell uname -s)
endif
# Get the arch-name if empty
ifeq ($(arch-name),)
	arch-name := $(shell uname -p)
endif

#########################
## Library Directories ##
#########################

BLAM_PATH := $(dir $(lastword $(MAKEFILE_LIST)))

CUDA_PATH = /usr/local/cuda

INTEL_PATH = /opt/intel
MKL_PATH = $(INTEL_PATH)/mkl

# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
BLAS_INC = -I$(MKL_PATH)/include
ifeq ($(kernel-name),Darwin)
	BLAS_LIBS = -lpthread -ldl -lm -L$(MKL_PATH)/lib -lmkl_rt -lmkl_intel_thread -L$(INTEL_PATH)/lib -liomp5 -lcublas
endif
ifeq ($(kernel-name),Linux)
	BLAS_LIBS =  -Wl,--start-group $(MKL_PATH)/lib/intel64/libmkl_intel_lp64.a $(MKL_PATH)/lib/intel64/libmkl_intel_thread.a $(MKL_PATH)/lib/intel64/libmkl_core.a /opt/intel/lib/intel64/libiomp5.a -Wl,--end-group -lpthread -ldl -lm -L$(MKL_PATH)/lib/intel64 -lcublas
endif

###############
## Compilers ##
###############

# Define the C++ compiler to use
ifeq ($(kernel-name),Darwin)
	USE_CXX := $(shell which g++)
endif
ifeq ($(kernel-name),Linux)
	USE_CXX := $(shell which g++-4.8)
endif
CXX := $(USE_CXX) -std=c++11
# Check for CUDA compiler
USE_NVCC := $(shell which nvcc)
NVCC     := $(USE_NVCC) -std=c++11 -ccbin=$(USE_CXX)

# Dependency directory and flags
DEPSDIR := $(shell mkdir -p .deps; echo .deps)
# MD: Dependency as side-effect of compilation
# MF: File for output
# MP: Include phony targets
DEPSFILE = $(DEPSDIR)/$(notdir $*.d)
DEPSFLAGS = -MD -MF $(DEPSFILE) #-MP

# Define any directories containing header files
#   To include directories use -Ipath/to/files
INCLUDES += -I$(BLAM_PATH) $(BLAS_INC)

# Define cxx compile flags
CXXFLAGS  = -O3 -fopenmp

# Define nvcc compile flags   TODO: Detect and generate appropriate sm_XX?
NVCCFLAGS := -arch=sm_35 -O3 --compiler-options "$(CXXFLAGS)"

# Define any directories containing libraries
#   To include directories use -Lpath/to/files
LDFLAGS +=

# Define any libraries to link into executable
#   To link in libraries (libXXX.so or libXXX.a) use -lXXX
LDLIBS  += $(BLAS_LIBS)

######################
## Makefile Options ##
######################

ifeq ($(NDEBUG),1)
	CXXFLAGS += -DNDEBUG
	NVCCFLAGS += -DNDEBUG
endif
ifeq ($(DEBUG),1)
	CXXFLAGS += -g -fno-inline
endif
ifeq ($(PROFILE),1)
	CXXFLAGS += -g -pg
endif
# Set up for CUDA if available
ifeq ($(USE_NVCC),)    # NVCC is not available
else                   # NVCC is available
	INCLUDES += -I$(CUDA_PATH)/include
	# Use cuda lib64 if it exists, else cuda lib
	ifneq ($(wildcard $(CUDA_PATH)/lib64/.*),)
		LDFLAGS += -L$(CUDA_PATH)/lib64
	else
		LDFLAGS += -L$(CUDA_PATH)/lib
	endif
	LDLIBS += -lcudart
endif

####################
## Makefile Rules ##
####################

# Suffix replacement rules
#   $^: the name of the prereqs of the rule
#   $<: the name of the first prereq of the rule
#   $@: the name of the target of the rule

# 'make' - default rule
all: $(EXEC)

# Default rule for creating an exec of $(EXEC) from a .o file
$(EXEC): % : %.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Default rule for creating a .o file from a .cpp file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPSFLAGS) -c -o $@ $<

# Default rule for creating a .o file from a .cu file
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -M -o $(DEPSFILE) $<

# 'make clean' - deletes all .o and temp files, exec, and dependency file
clean:
	-$(RM) *.o $(EXEC)
	$(RM) -r $(DEPSDIR)

# Define rules that do not actually generate the corresponding file
.PHONY: clean all

# Include the dependency files
-include $(wildcard $(DEPSDIR)/*.d)
