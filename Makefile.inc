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

BLAM_PATH := $(dir $(lastword ${MAKEFILE_LIST}))

include ${BLAM_PATH}/make.inc.mkl
#include $(BLAM_PATH)/make.inc.openblas
include ${BLAM_PATH}/make.inc.cublas

# defaults if not defined in make.inc
CXX      ?= g++

LDFLAGS  += -fPIC

CXXFLAGS += -O3 -std=c++11 -fPIC -pedantic \
            -Wall -Wno-unused-local-typedefs -Wno-unused-but-set-variable
#CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion
CXXFLAGS += -I${BLAM_PATH}

LIBS     ?= -lblas

# Dependency directory and flags
DEPSDIR := ${shell mkdir -p .deps; echo .deps}
# MD: Dependency as side-effect of compilation
# MF: File for output
# MP: Include phony targets
DEPSFILE = ${DEPSDIR}/$(notdir $*.d)
DEPSFLAGS = -MD -MF ${DEPSFILE} #-MP

####################
## Makefile Rules ##
####################

# Suffix replacement rules
#   $^: the name of the prereqs of the rule
#   $<: the name of the first prereq of the rule
#   $@: the name of the target of the rule

# 'make' - default rule
all: ${EXEC}

# Default rule for creating an exec of ${EXEC} from a .o file
${EXEC}: % : %.o
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS}

# Default rule for creating a .o file from a .cpp file
%.o: %.cpp
	${CXX} ${CXXFLAGS} ${INCLUDES} ${DEPSFLAGS} -c -o $@ $<

# Default rule for creating a .o file from a .cc file
%.o: %.cc
	${CXX} ${CXXFLAGS} ${INCLUDES} ${DEPSFLAGS} -c -o $@ $<

# Default rule for creating a .o file from a .cu file
%.o: %.cu
	${NVCC} -ccbin=${CXX} ${NVCCFLAGS} --compiler-options "${CXXFLAGS}" -c -o $@ $<
	@${NVCC} -ccbin=${CXX} ${NVCCFLAGS} --compiler-options "${CXXFLAGS}" -M -o ${DEPSFILE} $<

# 'make clean' - deletes all .o and temp files, exec, and dependency file
clean:
	-${RM} *.o ${EXEC}
	${RM} -r ${DEPSDIR}

# Define rules that do not actually generate the corresponding file
.PHONY: clean all

# Include the dependency files
-include ${wildcard ${DEPSDIR}/*.d}
