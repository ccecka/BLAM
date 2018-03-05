include ../Makefile.inc

# defaults if not defined in make.inc
CXX      ?= g++

LDFLAGS  ?= -fPIC -fopenmp
CXXFLAGS ?= -fPIC -fopenmp -MMD -std=c++11 -pedantic \
            -Wall -Wno-unused-local-typedefs -Wno-unused-but-set-variable \
            -I${CBLASDIR}
#CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion

LIBS     ?= -lblas

# ------------------------------------------------------------------------------
# BLAS++ specific flags
pwd = ${shell pwd}
libtest_path = ${realpath ${pwd}/../libtest}
libtest_src  = ${wildcard ${libtest_path}/*.cc} \
               ${wildcard ${libtest_path}/*.hh}
libtest_so   = ${libtest_path}/libtest.so

CXXFLAGS += -I../libtest

BLASPP_LIBS = -L../libtest -Wl,-rpath,${libtest_path} -ltest

# ------------------------------------------------------------------------------
# files
test_src = ${wildcard test_*.cc}
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

# ------------------------------------------------------------------------------
# rules
.PHONY: default all include

default: test

all: test

test: test.o ${test_obj} ${libtest_so}
	${CXX} ${LDFLAGS} -o $@ test.o ${test_obj} ${BLASPP_LIBS} ${LIBS}

${libtest_so}: ${libtest_src}
	cd ${libtest_path} && ${MAKE}
