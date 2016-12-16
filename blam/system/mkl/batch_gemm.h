/******************************************************************************
 * Copyright (C) 2016, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016, NVIDIA CORPORATION.  All rights reserved.
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

#if (INTEL_MKL_VERSION >= 110300)

#include <blam/detail/config.h>
#include <blam/system/mkl/execution_policy.h>

namespace blam
{
namespace mkl
{

// sgemm
void
batch_gemm(const CBLAS_LAYOUT order,
           const CBLAS_TRANSPOSE* transA, const CBLAS_TRANSPOSE* transB,
           const int* m, const int* n, const int* k,
           const float* alpha,
           const float** A_array, const int* ldA,
           const float** B_array, const int* ldB,
           const float* beta,
           float** C_array, int* ldC,
           const int group_count, const int* group_size)
{
  BLAM_DEBUG_OUT("cblas_sgemm_batch");

  cblas_sgemm_batch(order, transA, transB,
                    m, n, k,
                    alpha,
                    A_array, ldA,
                    B_array, ldB,
                    beta,
                    C_array, ldC,
                    group_count, group_size);
}

// dgemm
void
batch_gemm(const CBLAS_LAYOUT order,
           const CBLAS_TRANSPOSE* transA, const CBLAS_TRANSPOSE* transB,
           const int* m, const int* n, const int* k,
           const double* alpha,
           const double** A_array, const int* ldA,
           const double** B_array, const int* ldB,
           const double* beta,
           double** C_array, int* ldC,
           const int group_count, const int* group_size)
{
  BLAM_DEBUG_OUT("cblas_dgemm_batch");

  cblas_dgemm_batch(order, transA, transB,
                    m, n, k,
                    alpha,
                    A_array, ldA,
                    B_array, ldB,
                    beta,
                    C_array, ldC,
                    group_count, group_size);
}

// cgemm
void
batch_gemm(const CBLAS_LAYOUT order,
           const CBLAS_TRANSPOSE* transA, const CBLAS_TRANSPOSE* transB,
           const int* m, const int* n, const int* k,
           const ComplexFloat* alpha,
           const ComplexFloat** A_array, const int* ldA,
           const ComplexFloat** B_array, const int* ldB,
           const ComplexFloat* beta,
           ComplexFloat** C_array, int* ldC,
           const int group_count, const int* group_size)
{
  BLAM_DEBUG_OUT("cblas_cgemm_batch");

  cblas_cgemm_batch(order, transA, transB,
                    m, n, k,
                    reinterpret_cast<const void*>(alpha),
                    reinterpret_cast<const void**>(A_array), ldA,
                    reinterpret_cast<const void**>(B_array), ldB,
                    reinterpret_cast<const void*>(beta),
                    reinterpret_cast<void**>(C_array), ldC,
                    group_count, group_size);
}

// zgemm
void
batch_gemm(const CBLAS_LAYOUT order,
           const CBLAS_TRANSPOSE* transA, const CBLAS_TRANSPOSE* transB,
           const int* m, const int* n, const int* k,
           const ComplexDouble* alpha,
           const ComplexDouble** A_array, const int* ldA,
           const ComplexDouble** B_array, const int* ldB,
           const ComplexDouble* beta,
           ComplexDouble** C_array, int* ldC,
           const int group_count, const int* group_size)
{
  BLAM_DEBUG_OUT("cblas_zgemm_batch");

  cblas_zgemm_batch(order, transA, transB,
                    m, n, k,
                    reinterpret_cast<const void*>(alpha),
                    reinterpret_cast<const void**>(A_array), ldA,
                    reinterpret_cast<const void**>(B_array), ldB,
                    reinterpret_cast<const void*>(beta),
                    reinterpret_cast<void**>(C_array), ldC,
                    group_count, group_size);
}

// blam -> mkl
template <typename DerivedPolicy,
          typename T>
void
batch_gemm(const execution_policy<DerivedPolicy>& /*exec*/,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const T& alpha,
           const T* A, int ldA, int loA,
           const T* B, int ldB, int loB,
           const T& beta,
           T* C, int ldC, int loC,
           int batch_size)
{
  const T* a_array[batch_size];
  const T* b_array[batch_size];
  T* c_array[batch_size];
  for (int i = 0; i < batch_size; ++i) {
    a_array[i] = A + i*loA;
    b_array[i] = B + i*loB;
    c_array[i] = C + i*loC;
  }
  CBLAS_TRANSPOSE tA = cblas_transpose(transA);
  CBLAS_TRANSPOSE tB = cblas_transpose(transB);

  return batch_gemm(cblas_order(order), &tA, &tB,
                    &m, &n, &k,
                    &alpha,
                    a_array, &ldA,
                    b_array, &ldB,
                    &beta,
                    c_array, &ldC,
                    1, &batch_size);
}

} // end namespace mkl
} // end namespace blam

#endif // INTEL_MKL_VERSION >= 110300
