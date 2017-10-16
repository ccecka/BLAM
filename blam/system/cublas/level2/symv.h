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

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// ssymv
cublasStatus_t
symv(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const float* alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float* beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasSsymv");

  return cublasSsymv(handle, uplo,
                     n,
                     alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY);
}

// dsymv
cublasStatus_t
symv(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const double* alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double* beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDsymv");

  return cublasDsymv(handle, uplo,
                     n,
                     alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY);
}

// csymv
cublasStatus_t
symv(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat* beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCsymv");

  return cublasCsymv(handle, uplo,
                     n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(A), ldA,
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<const cuFloatComplex*>(beta),
                     reinterpret_cast<cuFloatComplex*>(y), incY);
}

// zsymv
cublasStatus_t
symv(cublasHandle_t handle, cublasFillMode_t uplo,
     int n,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble* beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZsymv");

  return cublasZsymv(handle, uplo,
                     n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<const cuDoubleComplex*>(beta),
                     reinterpret_cast<cuDoubleComplex*>(y), incY);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
symv(const execution_policy<DerivedPolicy>& exec,
     Uplo uplo,
     int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(symv(handle(derived_cast(exec)), cublas_type(uplo),
                     n,
                     &alpha,
                     A, ldA,
                     x, incX,
                     &beta,
                     y, incY))
{
  return symv(handle(derived_cast(exec)), cublas_type(uplo),
              n,
              &alpha,
              A, ldA,
              x, incX,
              &beta,
              y, incY);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
symv(const execution_policy<DerivedPolicy>& exec,
     Layout order, Uplo uplo,
     int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(symv(exec, uplo,
                     n, alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY))
{
  if (order == RowMajor) {
    // Swap upper <=> lower
    uplo = (uplo==Upper) ? Lower : Upper;
  }

  return symv(exec, uplo,
              n, alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

} // end namespace cublas
} // end namespace blam
