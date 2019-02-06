/******************************************************************************
 * Copyright (C) 2016-2019, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <blam/system/cublas/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sdgmm
inline cublasStatus_t
dgmm(cublasHandle_t handle,
     cublasSideMode_t mode,
     int m, int n,
     const float* A, int ldA,
     const float* x, int incX,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasSdgmm");

  return cublasSdgmm(handle, mode,
                     m, n,
                     A, ldA,
                     x, incX,
                     C, ldC);
}

// ddgmm
inline cublasStatus_t
dgmm(cublasHandle_t handle,
     cublasSideMode_t mode,
     int m, int n,
     const double* A, int ldA,
     const double* x, int incX,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDdgmm");

  return cublasDdgmm(handle, mode,
                     m, n,
                     A, ldA,
                     x, incX,
                     C, ldC);
}

// cdgmm
inline cublasStatus_t
dgmm(cublasHandle_t handle,
     cublasSideMode_t mode,
     int m, int n,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasCdgmm");

  return cublasCdgmm(handle, mode,
                     m, n,
                     reinterpret_cast<const cuFloatComplex*>(A), ldA,
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<cuFloatComplex*>(C), ldC);
}

// zdgmm
inline cublasStatus_t
dgmm(cublasHandle_t handle,
     cublasSideMode_t mode,
     int m, int n,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasZdgmm");

  return cublasZdgmm(handle, mode,
                     m, n,
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<cuDoubleComplex*>(C), ldC);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename MA, typename VX,
          typename MC>
inline auto
dgmm(const execution_policy<DerivedPolicy>& exec,
     Side mode,
     int m, int n,
     const MA* A, int ldA,
     const VX* x, int incX,
     MC* C, int ldC)
    -> decltype(dgmm(handle(derived_cast(exec)), cublas_type(mode),
                     m, n,
                     A, ldA,
                     x, incX,
                     C, ldC))
{
  return dgmm(handle(derived_cast(exec)), cublas_type(mode),
              m, n,
              A, ldA,
              x, incX,
              C, ldC);
}

// XXX TODO: RowMajor version?

} // end namespace cublas
} // end namespace blam
