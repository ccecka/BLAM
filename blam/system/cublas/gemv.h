#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename T>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose trans,
     int m, int n,
     const T& alpha,
     const T* A, int ldA,
     const T* x, int incX,
     const T& beta,
     T* y, int incY)
{
  if (order == ColMajor) {
    gemv(exec, trans,
         m, n,
         alpha,
         A, ldA,
         x, incX,
         beta,
         y, incY);
  } else { // RowMajor: transpose A
    gemv(exec, Transpose(trans ^ Trans),
         n, m,
         alpha,
         A, ldA,
         x, incX,
         beta,
         y, incY);
  }
}

// sgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     Transpose trans,
     int m, int n,
     const float& alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float& beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasSgemv");

  cublasSgemv(handle(derived_cast(exec)), cublas_transpose(trans),
              m, n,
              &alpha,
              A, ldA,
              x, incX,
              &beta,
              y, incY);
}

// dgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     Transpose trans,
     int m, int n,
     const double& alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double& beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDgemv");

  cublasDgemv(handle(derived_cast(exec)), cublas_transpose(trans),
              m, n,
              &alpha,
              A, ldA,
              x, incX,
              &beta,
              y, incY);
}

// cgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     Transpose trans,
     int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat& beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCgemv");

  cublasCgemv(handle(derived_cast(exec)), cublas_transpose(trans),
              m, n,
              reinterpret_cast<const cuFloatComplex*>(&alpha),
              reinterpret_cast<const cuFloatComplex*>(A), ldA,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(&beta),
              reinterpret_cast<cuFloatComplex*>(y), incY);
}

// zgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     Transpose trans,
     int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble& beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  cublasZgemv(handle(derived_cast(exec)), cublas_transpose(trans),
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(&alpha),
              reinterpret_cast<const cuDoubleComplex*>(A), ldA,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(&beta),
              reinterpret_cast<cuDoubleComplex*>(y), incY);
}

} // end namespace cublas
} // end namespace blam
