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
ger(const execution_policy<DerivedPolicy>& exec,
    StorageOrder order, int m, int n,
    const T& alpha,
    const T* x, int incX,
    const T* y, int incY,
    T* A, int ldA)
{
  if (order == ColMajor) {
    ger(exec, m, n,
        alpha,
        x, incX,
        y, incY,
        A, ldA);
  } else { // RowMajor: swap x & y
    ger(exec, n, m,
        alpha,
        y, incY,
        x, incX,
        A, ldA);
  }
}

// sger
template <typename DerivedPolicy>
void
ger(const execution_policy<DerivedPolicy>& exec,
    int m, int n,
    const float& alpha,
    const float* x, int incX,
    const float* y, int incY,
    float* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasSger");

  cublasSger(handle(derived_cast(exec)),
             m, n,
             &alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// dger
template <typename DerivedPolicy>
void
ger(const execution_policy<DerivedPolicy>& exec,
    int m, int n,
    const double& alpha,
    const double* x, int incX,
    const double* y, int incY,
    double* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasDger");

  cublasDger(handle(derived_cast(exec)),
             m, n,
             &alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// cgerc
template <typename DerivedPolicy>
void
ger(const execution_policy<DerivedPolicy>& exec,
    int m, int n,
    const ComplexFloat& alpha,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCgerc");

  cublasCgerc(handle(derived_cast(exec)),
              m, n,
              reinterpret_cast<const cuFloatComplex*>(&alpha),
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(y), incY,
              reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zgerc
template <typename DerivedPolicy>
void
ger(const execution_policy<DerivedPolicy>& exec,
    int m, int n,
    const ComplexDouble& alpha,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZgerc");

  cublasZgerc(handle(derived_cast(exec)),
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(&alpha),
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(y), incY,
              reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename T>
void
geru(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, int m, int n,
     const T& alpha,
     const T* x, int incX,
     const T* y, int incY,
     T* A, int ldA)
{
  if (order == ColMajor) {
    geru(exec, m, n,
         alpha,
         x, incX,
         y, incY,
         A, ldA);
  } else {
    geru(exec, n, m,
         alpha,
         y, incY,
         x, incX,
         A, ldA);
  }
}

// cgeru
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& exec,
     int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCgeru");

  cublasCgeru(handle(derived_cast(exec)),
              m, n,
              reinterpret_cast<const cuFloatComplex*>(&alpha),
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(y), incY,
              reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zgeru
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& exec,
     int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZgeru");

  cublasZgeru(handle(derived_cast(exec)),
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(&alpha),
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(y), incY,
              reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

} // end namespace cublas
} // end namespace blam
