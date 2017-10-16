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

// strsm
cublasStatus_t
trsm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t trans, cublasDiagType_t diag,
     int m, int n,
     const float* alpha,
     const float* A, int ldA,
     float* B, int ldB)
{
  BLAM_DEBUG_OUT("cublasStrsm");

  return cublasStrsm(handle, side, uplo, trans, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB);
}

// dtrsm
cublasStatus_t
trsm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t trans, cublasDiagType_t diag,
     int m, int n,
     const double* alpha,
     const double* A, int ldA,
     double* B, int ldB)
{
  BLAM_DEBUG_OUT("cublasDtrsm");

  return cublasDtrsm(handle, side, uplo, trans, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB);
}

// ctrsm
cublasStatus_t
trsm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t trans, cublasDiagType_t diag,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     ComplexFloat* B, int ldB)
{
  BLAM_DEBUG_OUT("cublasCtrsm");

  return cublasCtrsm(handle, side, uplo, trans, diag,
                     m, n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(A), ldA,
                     reinterpret_cast<cuFloatComplex*>(B), ldB);
}

// ztrsm
cublasStatus_t
trsm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t trans, cublasDiagType_t diag,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     ComplexDouble* B, int ldB)
{
  BLAM_DEBUG_OUT("cublasZtrsm");

  return cublasZtrsm(handle, side, uplo, trans, diag,
                     m, n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA,
                     reinterpret_cast<cuDoubleComplex*>(B), ldB);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB, typename MC>
auto
trsm(const execution_policy<DerivedPolicy>& exec,
     Side side, Uplo uplo, Op trans, Diag diag,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     MB* B, int ldB)
    -> decltype(trsm(handle(derived_cast(exec)),
                     cublas_type(side), cublas_type(uplo),
                     cublas_type(trans), cublas_type(diag),
                     m, n,
                     &alpha,
                     A, ldA,
                     B, ldB))
{
  return trsm(handle(derived_cast(exec)),
              cublas_type(side), cublas_type(uplo),
              cublas_type(trans), cublas_type(diag),
              m, n,
              &alpha,
              A, ldA,
              B, ldB);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB, typename MC>
auto
trsm(const execution_policy<DerivedPolicy>& exec,
     Layout order, Side side, Uplo uplo, Op trans, Diag diag,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     MB* B, int ldB)
    -> decltype(trsm(exec, side, uplo, trans, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB))
{
  if (order == RowMajor) {
    // Swap left <=> right, upper <=> lower
    side = (side==Left) ? Right : Left;
    uplo = (uplo==Upper) ? Lower : Upper;
  }

  return trsm(exec, side, uplo,
              trans, diag,
              m, n,
              alpha,
              A, ldA,
              B, ldB);
}

} // end namespace cublas
} // end namespace blam
