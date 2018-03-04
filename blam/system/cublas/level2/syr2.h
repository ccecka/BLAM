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

#include <blam/system/cublas/config.h>
#include <blam/system/cublas/execution_policy.h>

#include <blam/system/cublas/level3/syr2k.h>  // RowMajor

namespace blam
{
namespace cublas
{

// ssyr2
inline cublasStatus_t
syr2(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const float* alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasSsyr2");

  return cublasSsyr2(handle, uplo,
                     n,
                     alpha,
                     x, incX,
                     y, incY,
                     A, ldA);
}

// dsyr2
inline cublasStatus_t
syr2(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const double* alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasDsyr2");

  return cublasDsyr2(handle, uplo,
                     n,
                     alpha,
                     x, incX,
                     y, incY,
                     A, ldA);
}

// csyr2
inline cublasStatus_t
syr2(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const ComplexFloat* alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCsyr2");

  return cublasCsyr2(handle, uplo,
                     n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<const cuFloatComplex*>(y), incY,
                     reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zsyr2
inline cublasStatus_t
syr2(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const ComplexDouble* alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZsyr2");

  return cublasZsyr2(handle, uplo,
                     n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<const cuDoubleComplex*>(y), incY,
                     reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
inline auto
syr2(const execution_policy<DerivedPolicy>& exec,
     Uplo uplo,
     int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(syr2(handle(derived_cast(exec)), cublas_type(uplo),
                     n, &alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return syr2(handle(derived_cast(exec)), cublas_type(uplo),
              n, &alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
inline auto
syr2(const execution_policy<DerivedPolicy>& exec,
     Layout order, Uplo uplo,
     int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(syr2(exec, uplo,
                     n, alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  if (order == RowMajor) {
    if (std::is_same<MA,ComplexFloat>::value || std::is_same<MA,ComplexDouble>::value) {
      // No zero-overhead solution exists for RowMajor syr2. Options are:
      // 0) Fail with return code, assert, or throw
      // 1) Decay to many dot/axpy
      // 3) Promote to syr2k

      // Here, we've chosen (3), which works when incX > 0 and incY > 0
      // (Could consider a copy for incX < 0 and/or incY < 0)

      //assert(false && "No cublas::syr2 for RowMajor+Complex");
      //return CUBLAS_STATUS_INVALID_VALUE;

      MA beta = 1;
      return syr2k(exec, order, uplo, Trans,
                   n, 1,
                   alpha,
                   x, incX,
                   y, incY,
                   beta,
                   A, ldA);
    }


    // Swap upper <=> lower
    uplo = (uplo==Upper) ? Lower : Upper;
  }

  return syr2(exec, uplo,
              n, alpha,
              x, incX,
              y, incY,
              A, ldA);
}

} // end namespace cublas
} // end namespace blam
