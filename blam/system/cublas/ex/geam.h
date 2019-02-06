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

// sgeam
inline cublasStatus_t
geam(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n,
     const float* alpha,
     const float* A, int ldA,
     const float* beta,
     const float* B, int ldB,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasSgeam");

  return cublasSgeam(handle, transA, transB,
                     m, n,
                     alpha,
                     A, ldA,
                     beta,
                     B, ldB,
                     C, ldC);
}

// dgeam
inline cublasStatus_t
geam(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n,
     const double* alpha,
     const double* A, int ldA,
     const double* beta,
     const double* B, int ldB,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgeam");

  return cublasDgeam(handle, transA, transB,
                     m, n,
                     alpha,
                     A, ldA,
                     beta,
                     B, ldB,
                     C, ldC);
}

// cgeam
inline cublasStatus_t
geam(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* beta,
     const ComplexFloat* B, int ldB,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasCgeam");

  return cublasCgeam(handle, transA, transB,
                     m, n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(A), ldA,
                     reinterpret_cast<const cuFloatComplex*>(beta),
                     reinterpret_cast<const cuFloatComplex*>(B), ldB,
                     reinterpret_cast<cuFloatComplex*>(C), ldC);
}

// zgeam
inline cublasStatus_t
geam(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* beta,
     const ComplexDouble* B, int ldB,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasZgeam");

  return cublasZgeam(handle, transA, transB,
                     m, n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex*>(beta),
                     reinterpret_cast<const cuDoubleComplex*>(B), ldB,
                     reinterpret_cast<cuDoubleComplex*>(C), ldC);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
inline auto
geam(const execution_policy<DerivedPolicy>& exec,
     Op transA, Op transB,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const Beta& beta,
     const MB* B, int ldB,
     MC* C, int ldC)
    -> decltype(geam(handle(derived_cast(exec)),
                     cublas_type(transA), cublas_type(transB),
                     m, n,
                     &alpha,
                     A, ldA,
                     &beta,
                     B, ldB,
                     C, ldC))
{
  return geam(handle(derived_cast(exec)),
              cublas_type(transA), cublas_type(transB),
              m, n,
              &alpha,
              A, ldA,
              &beta,
              B, ldB,
              C, ldC);
}

// XXX TODO: RowMajor version?

} // end namespace cublas
} // end namespace blam
