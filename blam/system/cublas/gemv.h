#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose trans,
     int m, int n,
     const float& alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float& beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasSgemv");

  if (order == RowMajor) {
    trans = Transpose(trans^Trans);
    using std::swap;
    swap(m, n);
  }

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
     StorageOrder order, Transpose trans,
     int m, int n,
     const double& alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double& beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDgemv");

  if (order == RowMajor) {
    trans = Transpose(trans^Trans);
    using std::swap;
    swap(m, n);
  }

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
     StorageOrder order, Transpose trans,
     int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat& beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCgemv");

  if (order == RowMajor) {
    trans = Transpose(trans^Trans);
    using std::swap;
    swap(m, n);
  }

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
     StorageOrder order, Transpose trans,
     int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble& beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  if (order == RowMajor) {
    trans = Transpose(trans^Trans);
    using std::swap;
    swap(m, n);
  }

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
