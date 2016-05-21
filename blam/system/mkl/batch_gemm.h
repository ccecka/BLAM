#pragma once

// XXX: Conditional on MKL 11.3beta

#include <blam/detail/config.h>
#include <blam/system/mkl/execution_policy.h>

namespace blam
{
namespace mkl
{

// sgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& /*exec*/,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const float& alpha,
           const float* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float& beta,
           float* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cblas_sgemm_batch");

  const float* a_array[p];
  const float* b_array[p];
  float* c_array[p];
  for (int i = 0; i < p; ++i) {
    a_array[i] = A + i*loA;
    b_array[i] = B + i*loB;
    c_array[i] = C + i*loC;
  }
  CBLAS_TRANSPOSE tA = cblas_transpose(transA);
  CBLAS_TRANSPOSE tB = cblas_transpose(transB);

  cblas_sgemm_batch(cblas_order(order),
                    &tA, &tB,
                    &m, &n, &k,
                    &alpha,
                    a_array, &ldA,
                    b_array, &ldB,
                    &beta,
                    c_array, &ldC,
                    1, &p);
}

// dgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& /*exec*/,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const double& alpha,
           const double* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double& beta,
           double* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cblas_dgemm_batch");

  const double* a_array[p];
  const double* b_array[p];
  double* c_array[p];
  for (int i = 0; i < p; ++i) {
    a_array[i] = A + i*loA;
    b_array[i] = B + i*loB;
    c_array[i] = C + i*loC;
  }
  CBLAS_TRANSPOSE tA = cblas_transpose(transA);
  CBLAS_TRANSPOSE tB = cblas_transpose(transB);

  cblas_dgemm_batch(cblas_order(order),
                    &tA, &tB,
                    &m, &n, &k,
                    &alpha,
                    a_array, &ldA,
                    b_array, &ldB,
                    &beta,
                    c_array, &ldC,
                    1, &p);
}

// cgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& /*exec*/,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const ComplexFloat& alpha,
           const ComplexFloat* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const ComplexFloat& beta,
           ComplexFloat* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cblas_cgemm_batch");

  const ComplexFloat* a_array[p];
  const ComplexFloat* b_array[p];
  ComplexFloat* c_array[p];
  for (int i = 0; i < p; ++i) {
    a_array[i] = A + i*loA;
    b_array[i] = B + i*loB;
    c_array[i] = C + i*loC;
  }
  CBLAS_TRANSPOSE tA = cblas_transpose(transA);
  CBLAS_TRANSPOSE tB = cblas_transpose(transB);

  cblas_cgemm_batch(cblas_order(order),
                    &tA, &tB,
                    &m, &n, &k,
                    reinterpret_cast<const float*>(&alpha),
                    reinterpret_cast<const void**>(a_array), &ldA,
                    reinterpret_cast<const void**>(b_array), &ldB,
                    reinterpret_cast<const float*>(&beta),
                    reinterpret_cast<void**>(c_array), &ldC,
                    1, &p);
}

// zgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& /*exec*/,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const ComplexDouble& alpha,
           const ComplexDouble* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const ComplexDouble& beta,
           ComplexDouble* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cblas_zgemm_batch");

  const ComplexDouble* a_array[p];
  const ComplexDouble* b_array[p];
  ComplexDouble* c_array[p];
  for (int i = 0; i < p; ++i) {
    a_array[i] = A + i*loA;
    b_array[i] = B + i*loB;
    c_array[i] = C + i*loC;
  }
  CBLAS_TRANSPOSE tA = cblas_transpose(transA);
  CBLAS_TRANSPOSE tB = cblas_transpose(transB);

  cblas_zgemm_batch(cblas_order(order),
                    &tA, &tB,
                    &m, &n, &k,
                    reinterpret_cast<const double*>(&alpha),
                    reinterpret_cast<const void**>(a_array), &ldA,
                    reinterpret_cast<const void**>(b_array), &ldB,
                    reinterpret_cast<const double*>(&beta),
                    reinterpret_cast<void**>(c_array), &ldC,
                    1, &p);
}

} // end namespace mkl
} // end namespace blam
