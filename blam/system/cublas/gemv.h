#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const float* alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float* beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasSgemv");

  cublasSgemv(handle, trans,
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// dgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const double* alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double* beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDgemv");

  cublasDgemv(handle, trans,
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// cgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat* beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCgemv");

  cublasCgemv(handle, trans,
              m, n,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<const cuFloatComplex*>(A), ldA,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(beta),
              reinterpret_cast<cuFloatComplex*>(y), incY);
}

// zgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble* beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  cublasZgemv(handle, trans,
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<const cuDoubleComplex*>(A), ldA,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(beta),
              reinterpret_cast<cuDoubleComplex*>(y), incY);
}

// csgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const float* alpha,
     const ComplexFloat* A, int ldA,
     const float* x, int incX,
     const float* beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  assert(incY == 1);

  cublasSgemv(handle, trans,
              2*m, n,
              reinterpret_cast<const float*>(alpha),
              reinterpret_cast<const float*>(A), 2*ldA,
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(beta),
              reinterpret_cast<float*>(y), incY);
}

// zdgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const double* alpha,
     const ComplexDouble* A, int ldA,
     const double* x, int incX,
     const double* beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  assert(incY == 1);

  cublasDgemv(handle, trans,
              2*m, n,
              reinterpret_cast<const double*>(alpha),
              reinterpret_cast<const double*>(A), 2*ldA,
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(beta),
              reinterpret_cast<double*>(y), incY);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename T>
void
gemv(const execution_policy<DerivedPolicy>& exec,
     Transpose trans,
     int m, int n,
     const T& alpha,
     const T* A, int ldA,
     const T* x, int incX,
     const T& beta,
     T* y, int incY)
{
  gemv(handle(derived_cast(exec)), cublas_transpose(trans),
       m, n,
       &alpha,
       A, ldA,
       x, incX,
       &beta,
       y, incY);
}

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

} // end namespace cublas
} // end namespace blam
