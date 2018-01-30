/******************************************************************************
 * Copyright (C) 2016-2017, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#pragma once

// Execution policy first
#include <blam/system/cublas/execution_policy.h>

// Include all algorithms

// Level 1
#include <blam/system/cublas/level1/asum.h>
#include <blam/system/cublas/level1/axpy.h>
#include <blam/system/cublas/level1/nrm2.h>
#include <blam/system/cublas/level1/copy.h>
#include <blam/system/cublas/level1/dot.h>
#include <blam/system/cublas/level1/swap.h>
#include <blam/system/cublas/level1/scal.h>
#include <blam/system/cublas/level1/iamax.h>

// Level 2
#include <blam/system/cublas/level2/spr.h>
#include <blam/system/cublas/level2/gemv.h>
#include <blam/system/cublas/level2/symv.h>
#include <blam/system/cublas/level2/tpmv.h>
#include <blam/system/cublas/level2/tpsv.h>
#include <blam/system/cublas/level2/spr2.h>
#include <blam/system/cublas/level2/spmv.h>
#include <blam/system/cublas/level2/hbmv.h>
#include <blam/system/cublas/level2/tbmv.h>
#include <blam/system/cublas/level2/her.h>
#include <blam/system/cublas/level2/hpmv.h>
#include <blam/system/cublas/level2/sbmv.h>
#include <blam/system/cublas/level2/tbsv.h>
#include <blam/system/cublas/level2/her2.h>
#include <blam/system/cublas/level2/hpr.h>
#include <blam/system/cublas/level2/hemv.h>
#include <blam/system/cublas/level2/trmv.h>
#include <blam/system/cublas/level2/gbmv.h>
#include <blam/system/cublas/level2/syr2.h>
#include <blam/system/cublas/level2/hpr2.h>
#include <blam/system/cublas/level2/trsv.h>
#include <blam/system/cublas/level2/syr.h>
#include <blam/system/cublas/level2/ger.h>

// Level 3
#include <blam/system/cublas/level3/gemm_batch.h>
#include <blam/system/cublas/level3/gemm.h>
#include <blam/system/cublas/level3/hemm.h>
#include <blam/system/cublas/level3/her2k.h>
#include <blam/system/cublas/level3/syr2k.h>
#include <blam/system/cublas/level3/herk.h>
#include <blam/system/cublas/level3/trmm.h>
#include <blam/system/cublas/level3/trsm.h>
#include <blam/system/cublas/level3/syrk.h>
#include <blam/system/cublas/level3/symm.h>

// Extended
#include <blam/system/cublas/ex/geam.h>
