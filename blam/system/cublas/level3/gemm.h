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

// sgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float* beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasSgemm");

  cublasSgemm(handle, transA, transB,
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// dgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const double* alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     const double* beta,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgemm");

  cublasDgemm(handle, transA, transB,
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// cgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     const ComplexFloat* beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasCgemm");

  cublasCgemm(handle, transA, transB,
              m, n, k,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<const cuFloatComplex*>(A), ldA,
              reinterpret_cast<const cuFloatComplex*>(B), ldB,
              reinterpret_cast<const cuFloatComplex*>(beta),
              reinterpret_cast<cuFloatComplex*>(C), ldC);
}

// zgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     const ComplexDouble* beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgemm");

  cublasZgemm(handle, transA, transB,
              m, n, k,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<const cuDoubleComplex*>(A), ldA,
              reinterpret_cast<const cuDoubleComplex*>(B), ldB,
              reinterpret_cast<const cuDoubleComplex*>(beta),
              reinterpret_cast<cuDoubleComplex*>(C), ldC);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
auto
gemm(const execution_policy<DerivedPolicy>& exec,
     Op transA, Op transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
    -> decltype(gemm(handle(derived_cast(exec)),
                     cublas_type(transA), cublas_type(transB),
                     m, n, k,
                     &alpha,
                     A, ldA,
                     B, ldB,
                     &beta,
                     C, ldC))
{
  return gemm(handle(derived_cast(exec)),
              cublas_type(transA), cublas_type(transB),
              m, n, k,
              &alpha,
              A, ldA,
              B, ldB,
              &beta,
              C, ldC);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
auto
gemm(const execution_policy<DerivedPolicy>& exec,
     Layout order, Op transA, Op transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
    -> decltype(gemm(exec, transA, transB,
                     m, n, k,
                     &alpha,
                     A, ldA,
                     B, ldB,
                     &beta,
                     C, ldC),
                gemm(exec, transB, transA,
                     n, m, k,
                     &alpha,
                     B, ldB,
                     A, ldA,
                     &beta,
                     C, ldC))
{
  if (order == ColMajor) {
    return gemm(exec, transA, transB,
                m, n, k,
                &alpha,
                A, ldA,
                B, ldB,
                &beta,
                C, ldC);
  } else { // RowMajor: swap A & B
    return gemm(exec, transB, transA,
                n, m, k,
                &alpha,
                B, ldB,
                A, ldA,
                &beta,
                C, ldC);
  }
}

} // end namespace cublas
} // end namespace blam
