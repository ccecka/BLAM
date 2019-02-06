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

// hgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const __half* alpha,
           const __half* A, int ldA, int loA,
           const __half* B, int ldB, int loB,
           const __half* beta,
           __half* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasHgemmStridedBatched");

  return cublasHgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   alpha,
                                   A, ldA, loA,
                                   B, ldB, loB,
                                   beta,
                                   C, ldC, loC,
                                   batch_size);
}

// sgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float* beta,
           float* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasSgemmStridedBatched");

  return cublasSgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   alpha,
                                   A, ldA, loA,
                                   B, ldB, loB,
                                   beta,
                                   C, ldC, loC,
                                   batch_size);
}

// dgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double* beta,
           double* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasDgemmStridedBatched");

  return cublasDgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   alpha,
                                   A, ldA, loA,
                                   B, ldB, loB,
                                   beta,
                                   C, ldC, loC,
                                   batch_size);
}

// cgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const ComplexFloat* alpha,
           const ComplexFloat* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const ComplexFloat* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasCgemmStridedBatched");

  return cublasCgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   reinterpret_cast<const cuFloatComplex*>(alpha),
                                   reinterpret_cast<const cuFloatComplex*>(A), ldA, loA,
                                   reinterpret_cast<const cuFloatComplex*>(B), ldB, loB,
                                   reinterpret_cast<const cuFloatComplex*>(beta),
                                   reinterpret_cast<cuFloatComplex*>(C), ldC, loC,
                                   batch_size);
}

// zgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const ComplexDouble* alpha,
           const ComplexDouble* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const ComplexDouble* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasZgemmStridedBatched");

  return cublasZgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   reinterpret_cast<const cuDoubleComplex*>(alpha),
                                   reinterpret_cast<const cuDoubleComplex*>(A), ldA, loA,
                                   reinterpret_cast<const cuDoubleComplex*>(B), ldB, loB,
                                   reinterpret_cast<const cuDoubleComplex*>(beta),
                                   reinterpret_cast<cuDoubleComplex*>(C), ldC, loC,
                                   batch_size);
}

// hgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const __half* alpha,
           const __half* const A[], int ldA,
           const __half* const B[], int ldB,
           const __half* beta,
           __half* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasHgemmBatched");

  return cublasHgemmBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            const_cast<const __half**>(A), ldA,
                            // A, ldA,   // cuBLAS 9.2
                            const_cast<const __half**>(B), ldB,
                            // B, ldB,   // cuBLAS 9.2
                            beta,
                            const_cast<__half**>(C), ldC,
                            // C, ldC,   // cuBLAS 9.2
                            batch_size);
}

// sgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* const A[], int ldA,
           const float* const B[], int ldB,
           const float* beta,
           float* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasSgemmBatched");

  return cublasSgemmBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            const_cast<const float**>(A), ldA,
                            // A, ldA,   // cuBLAS 9.2
                            const_cast<const float**>(B), ldB,
                            // B, ldB,   // cuBLAS 9.2
                            beta,
                            const_cast<float**>(C), ldC,
                            // C, ldC,   // cuBLAS 9.2
                            batch_size);
}

// dgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* const A[], int ldA,
           const double* const B[], int ldB,
           const double* beta,
           double* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasDgemmBatched");

  return cublasDgemmBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            const_cast<const double**>(A), ldA,
                            // A, ldA,   // cuBLAS 9.2
                            const_cast<const double**>(B), ldB,
                            // B, ldB,   // cuBLAS 9.2
                            beta,
                            const_cast<double**>(C), ldC,
                            // C, ldC,   // cuBLAS 9.2
                            batch_size);
}

// cgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const ComplexFloat* alpha,
           const ComplexFloat* const A[], int ldA,
           const ComplexFloat* const B[], int ldB,
           const ComplexFloat* beta,
           ComplexFloat* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasCgemmBatched");

  return cublasCgemmBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const cuFloatComplex*>(alpha),
                            const_cast<const cuFloatComplex**>(reinterpret_cast<const cuFloatComplex* const *>(A)), ldA,
                            //reinterpret_cast<const cuFloatComplex* const *>(A), ldA,  // cuBLAS 9.2
                            const_cast<const cuFloatComplex**>(reinterpret_cast<const cuFloatComplex* const *>(B)), ldB,
                            //reinterpret_cast<const cuFloatComplex* const *>(B), ldB,  // cuBLAS 9.2
                            reinterpret_cast<const cuFloatComplex*>(beta),
                            const_cast<cuFloatComplex**>(reinterpret_cast<cuFloatComplex* const *>(C)), ldC,
                            //reinterpret_cast<cuFloatComplex* const *>(C), ldC,        // cuBLAS 9.2
                            batch_size);
}

// zgemm
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const ComplexDouble* alpha,
           const ComplexDouble* const A[], int ldA,
           const ComplexDouble* const B[], int ldB,
           const ComplexDouble* beta,
           ComplexDouble* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasZgemmBatched");

  return cublasZgemmBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const cuDoubleComplex*>(alpha),
                            const_cast<const cuDoubleComplex**>(reinterpret_cast<const cuDoubleComplex* const *>(A)), ldA,
                            //reinterpret_cast<const cuDoubleComplex* const *>(A), ldA,  // cuBLAS 9.2
                            const_cast<const cuDoubleComplex**>(reinterpret_cast<const cuDoubleComplex* const *>(B)), ldB,
                            //reinterpret_cast<const cuDoubleComplex* const *>(B), ldB,  // cuBLAS 9.2
                            reinterpret_cast<const cuDoubleComplex*>(beta),
                            const_cast<cuDoubleComplex**>(reinterpret_cast<cuDoubleComplex* const *>(C)), ldC,
                            //reinterpret_cast<cuDoubleComplex* const *>(C), ldC,        // cuBLAS 9.2
                            batch_size);
}

// scgemm    XXX: Move to general?
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const ComplexFloat* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasS[C]gemmStridedBatched");

  assert(transA == CUBLAS_OP_N);

  return cublasSgemmStridedBatched(handle, transA, transB,
                                   2*m, n, k,
                                   reinterpret_cast<const float*>(alpha),
                                   reinterpret_cast<const float*>(A), 2*ldA, 2*loA,
                                   reinterpret_cast<const float*>(B), ldB, loB,
                                   reinterpret_cast<const float*>(beta),
                                   reinterpret_cast<float*>(C), 2*ldC, 2*loC,
                                   batch_size);
}

// csgemm    XXX: Move to general?
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const float* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasS[C]gemmStridedBatched");

  assert(transB == CUBLAS_OP_T);

  return cublasSgemmStridedBatched(handle, transA, transB,
                                   m, 2*n, k,
                                   reinterpret_cast<const float*>(alpha),
                                   reinterpret_cast<const float*>(A), ldA, loA,
                                   reinterpret_cast<const float*>(B), 2*ldB, 2*loB,
                                   reinterpret_cast<const float*>(beta),
                                   reinterpret_cast<float*>(C), 2*ldC, 2*loC,
                                   batch_size);
}

// zdgemm    XXX: Move to general?
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const ComplexDouble* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasD[Z]gemmStridedBatched");

  assert(transA == CUBLAS_OP_N);

  return cublasDgemmStridedBatched(handle, transA, transB,
                                   2*m, n, k,
                                   reinterpret_cast<const double*>(alpha),
                                   reinterpret_cast<const double*>(A), 2*ldA, 2*loA,
                                   reinterpret_cast<const double*>(B), ldB, loB,
                                   reinterpret_cast<const double*>(beta),
                                   reinterpret_cast<double*>(C), 2*ldC, 2*loC,
                                   batch_size);
}

// dzgemm    XXX: Move to general?
inline cublasStatus_t
gemm_batch(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const double* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasD[Z]gemmStridedBatched");

  assert(transB == CUBLAS_OP_T);

  return cublasDgemmStridedBatched(handle, transA, transB,
                                   m, 2*n, k,
                                   reinterpret_cast<const double*>(alpha),
                                   reinterpret_cast<const double*>(A), ldA, loA,
                                   reinterpret_cast<const double*>(B), 2*ldB, 2*loB,
                                   reinterpret_cast<const double*>(beta),
                                   reinterpret_cast<double*>(C), 2*ldC, 2*loC,
                                   batch_size);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
inline auto
gemm_batch(const execution_policy<DerivedPolicy>& exec,
           Op transA, Op transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int batch_size)
    -> decltype(gemm_batch(handle(derived_cast(exec)),
                           cublas_type(transA), cublas_type(transB),
                           m, n, k,
                           &alpha,
                           A, ldA, loA,
                           B, ldB, loB,
                           &beta,
                           C, ldC, loC,
                           batch_size))
{
  return gemm_batch(handle(derived_cast(exec)),
                    cublas_type(transA), cublas_type(transB),
                    m, n, k,
                    &alpha,
                    A, ldA, loA,
                    B, ldB, loB,
                    &beta,
                    C, ldC, loC,
                    batch_size);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
inline auto
gemm_batch(const execution_policy<DerivedPolicy>& exec,
           Layout order, Op transA, Op transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int batch_size)
    -> decltype(gemm_batch(exec, transA, transB,
                           m, n, k,
                           alpha,
                           A, ldA, loA,
                           B, ldB, loB,
                           beta,
                           C, ldC, loC,
                           batch_size))
{
  if (order == ColMajor) {
    return gemm_batch(exec, transA, transB,
                      m, n, k,
                      alpha,
                      A, ldA, loA,
                      B, ldB, loB,
                      beta,
                      C, ldC, loC,
                      batch_size);
  } else { // RowMajor: swap A <=> B, transA <=> transB, m <=> n
    return gemm_batch(exec, transB, transA,
                      n, m, k,
                      alpha,
                      B, ldB, loB,
                      A, ldA, loA,
                      beta,
                      C, ldC, loC,
                      batch_size);
  }
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
inline auto
gemm_batch(const execution_policy<DerivedPolicy>& exec,
           Op transA, Op transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* const A[], int ldA,
           const MB* const B[], int ldB,
           const Beta& beta,
           MC* const C[], int ldC,
           int batch_size)
    -> decltype(gemm_batch(handle(derived_cast(exec)),
                           cublas_type(transA), cublas_type(transB),
                           m, n, k,
                           &alpha,
                           A, ldA,
                           B, ldB,
                           &beta,
                           C, ldC,
                           batch_size))
{
  return gemm_batch(handle(derived_cast(exec)),
                    cublas_type(transA), cublas_type(transB),
                    m, n, k,
                    &alpha,
                    A, ldA,
                    B, ldB,
                    &beta,
                    C, ldC,
                    batch_size);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
inline auto
gemm_batch(const execution_policy<DerivedPolicy>& exec,
           Layout order, Op transA, Op transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* const A[], int ldA,
           const MB* const B[], int ldB,
           const Beta& beta,
           MC* const C[], int ldC,
           int batch_size)
    -> decltype(gemm_batch(exec, transA, transB,
                           m, n, k,
                           alpha,
                           A, ldA,
                           B, ldB,
                           beta,
                           C, ldC,
                           batch_size))
{
  if (order == ColMajor) {
    return gemm_batch(exec, transA, transB,
                      m, n, k,
                      alpha,
                      A, ldA,
                      B, ldB,
                      beta,
                      C, ldC,
                      batch_size);
  } else { // RowMajor: swap A <=> B, transA <=> transB, m <=> n
    return gemm_batch(exec, transB, transA,
                      n, m, k,
                      alpha,
                      B, ldB,
                      A, ldA,
                      beta,
                      C, ldC,
                      batch_size);
  }
}

} // end namespace cublas
} // end namespace blam
