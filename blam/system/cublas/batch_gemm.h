#pragma once

#if 0  // REPLACE WITH LIBRARY VERSIONING

#include <blam/detail/config.h>

#include <blam/system/cublas/execution_policy.h>

// XXX: Custom CUBLAS implementation of strided_batch_gemm
// XXX: Remove on CUBLAS 8.0
#include "cublas_batch_gemm.cuh"

namespace blam
{
namespace cublas
{

// sgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& exec,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const float& alpha,
           const float* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float& beta,
           float* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cublasSgemmBatched");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
    swap(loA, loB);
  }

  cublasSgemmBatched(handle(derived_cast(exec)),
                     cublas_transpose(transA), cublas_transpose(transB),
                     m, n, k,
                     &alpha,
                     A, ldA, loA,
                     B, ldB, loB,
                     &beta,
                     C, ldC, loC,
                     p);
}

// dgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& exec,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const double& alpha,
           const double* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double& beta,
           double* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cublasDgemmBatched");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
    swap(loA, loB);
  }

  cublasDgemmBatched(handle(derived_cast(exec)),
                     cublas_transpose(transA), cublas_transpose(transB),
                     m, n, k,
                     &alpha,
                     A, ldA, loA,
                     B, ldB, loB,
                     &beta,
                     C, ldC, loC,
                     p);
}

// cgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& exec,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const ComplexFloat& alpha,
           const ComplexFloat* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const ComplexFloat& beta,
           ComplexFloat* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cublasCgemmBatched");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
    swap(loA, loB);
  }

  cublasCgemmBatched(handle(derived_cast(exec)),
                     cublas_transpose(transA), cublas_transpose(transB),
                     m, n, k,
                     reinterpret_cast<const cuFloatComplex*>(&alpha),
                     reinterpret_cast<const cuFloatComplex*>(A), ldA, loA,
                     reinterpret_cast<const cuFloatComplex*>(B), ldB, loB,
                     reinterpret_cast<const cuFloatComplex*>(&beta),
                     reinterpret_cast<cuFloatComplex*>(C), ldC, loC,
                     p);
}

// zgemm
template <typename DerivedPolicy>
void
batch_gemm(const execution_policy<DerivedPolicy>& exec,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const ComplexDouble& alpha,
           const ComplexDouble* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const ComplexDouble& beta,
           ComplexDouble* C, int ldC, int loC,
           int p)
{
  BLAM_DEBUG_OUT("cublasZgemmBatched");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
    swap(loA, loB);
  }

  cublasZgemmBatched(handle(derived_cast(exec)),
                     cublas_transpose(transA), cublas_transpose(transB),
                     m, n, k,
                     reinterpret_cast<const cuDoubleComplex*>(&alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA, loA,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldB, loB,
                     reinterpret_cast<const cuDoubleComplex*>(&beta),
                     reinterpret_cast<cuDoubleComplex*>(C), ldC, loC,
                     p);
}

} // end namespace cublas
} // end namespace blam

#endif
