#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sgemm
template <typename DerivedPolicy>
void
gemm(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const float& alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float& beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasSgemm");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
  }

  cublasSgemm(handle(derived_cast(exec)),
              cublas_transpose(transA), cublas_transpose(transB),
              m, n, k,
              &alpha,
              A, ldA,
              B, ldB,
              &beta,
              C, ldC);
}

// dgemm
template <typename DerivedPolicy>
void
gemm(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const double& alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     const double& beta,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgemm");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
  }

  cublasDgemm(handle(derived_cast(exec)),
              cublas_transpose(transA), cublas_transpose(transB),
              m, n, k,
              &alpha,
              A, ldA,
              B, ldB,
              &beta,
              C, ldC);
}

// cgemm
template <typename DerivedPolicy>
void
gemm(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     const ComplexFloat& beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasCgemm");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
  }

  cublasCgemm(handle(derived_cast(exec)),
              cublas_transpose(transA), cublas_transpose(transB),
              m, n, k,
              reinterpret_cast<const cuFloatComplex*>(&alpha),
              reinterpret_cast<const cuFloatComplex*>(A), ldA,
              reinterpret_cast<const cuFloatComplex*>(B), ldB,
              reinterpret_cast<const cuFloatComplex*>(&beta),
              reinterpret_cast<cuFloatComplex*>(C), ldC);
}

// zgemm
template <typename DerivedPolicy>
void
gemm(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     const ComplexDouble& beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgemm");

  if (order == RowMajor) {
    using std::swap;
    swap(transA, transB);
    swap(m, n);
    swap(A, B);
    swap(ldA, ldB);
  }

  cublasZgemm(handle(derived_cast(exec)),
              cublas_transpose(transA), cublas_transpose(transB),
              m, n, k,
              reinterpret_cast<const cuDoubleComplex*>(&alpha),
              reinterpret_cast<const cuDoubleComplex*>(A), ldA,
              reinterpret_cast<const cuDoubleComplex*>(B), ldB,
              reinterpret_cast<const cuDoubleComplex*>(&beta),
              reinterpret_cast<cuDoubleComplex*>(C), ldC);
}

} // end namespace cublas
} // end namespace blam
