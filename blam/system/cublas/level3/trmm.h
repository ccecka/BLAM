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

// strmm
inline cublasStatus_t
trmm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t transA, cublasDiagType_t diag,
     int m, int n,
     const float* alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasStrmm");

  return cublasStrmm(handle, side, uplo, transA, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB,
                     C, ldC);
}

// dtrmm
inline cublasStatus_t
trmm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t transA, cublasDiagType_t diag,
     int m, int n,
     const double* alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDtrmm");

  return cublasDtrmm(handle, side, uplo, transA, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB,
                     C, ldC);
}

// ctrmm
inline cublasStatus_t
trmm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t transA, cublasDiagType_t diag,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasCtrmm");

  return cublasCtrmm(handle, side, uplo, transA, diag,
                     m, n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(A), ldA,
                     reinterpret_cast<const cuFloatComplex*>(B), ldB,
                     reinterpret_cast<cuFloatComplex*>(C), ldC);
}

// ztrmm
inline cublasStatus_t
trmm(cublasHandle_t handle,
     cublasSideMode_t side, cublasFillMode_t uplo,
     cublasOperation_t transA, cublasDiagType_t diag,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasZtrmm");

  return cublasZtrmm(handle, side, uplo, transA, diag,
                     m, n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldB,
                     reinterpret_cast<cuDoubleComplex*>(C), ldC);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB, typename MC>
inline auto
trmm(const execution_policy<DerivedPolicy>& exec,
     Side side, Uplo uplo, Op transA, Diag diag,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     MC* C, int ldC)
    -> decltype(trmm(handle(derived_cast(exec)),
                     cublas_type(side), cublas_type(uplo),
                     cublas_type(transA), cublas_type(diag),
                     m, n,
                     &alpha,
                     A, ldA,
                     B, ldB,
                     C, ldC))
{
  return trmm(handle(derived_cast(exec)),
              cublas_type(side), cublas_type(uplo),
              cublas_type(transA), cublas_type(diag),
              m, n,
              &alpha,
              A, ldA,
              B, ldB,
              C, ldC);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB, typename MC>
inline auto
trmm(const execution_policy<DerivedPolicy>& exec,
     Layout order, Side side, Uplo uplo, Op transA, Diag diag,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     MC* C, int ldC)
    -> decltype(trmm(exec, side, uplo, transA, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB,
                     C, ldC))
{
  if (order == RowMajor) {
    // Swap left <=> right, upper <=> lower, m <=> n
    side = (side==Left) ? Right : Left;
    uplo = (uplo==Upper) ? Lower : Upper;
    std::swap(m,n);
  }

  return trmm(exec, side, uplo, transA, diag,
              m, n,
              alpha,
              A, ldA,
              B, ldB,
              C, ldC);
}

// blam -> cublas: in-place
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB>
inline auto
trmm(const execution_policy<DerivedPolicy>& exec,
     Layout order, Side side, Uplo uplo, Op transA, Diag diag,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     MB* B, int ldB)
    -> decltype(trmm(exec, order, side, uplo, transA, diag,
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB,
                     B, ldB))
{
  return trmm(exec, order, side, uplo, transA, diag,
              m, n,
              alpha,
              A, ldA,
              B, ldB,
              B, ldB);
}

} // end namespace cublas
} // end namespace blam
