#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sger
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& exec,
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
geru(const execution_policy<DerivedPolicy>& exec,
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
gerc(const execution_policy<DerivedPolicy>& exec,
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
gerc(const execution_policy<DerivedPolicy>& exec,
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

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY, typename MA>
void
gerc(const execution_policy<DerivedPolicy>& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
{
  gerc(handle(derived_cast(exec)), n,
       x, incX,
       y, incY,
       A, ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename T>
void
gerc(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, int m, int n,
     const T& alpha,
     const T* x, int incX,
     const T* y, int incY,
     T* A, int ldA)
{
  if (order == ColMajor || (m == n && x == y && incX == incY)) {
    gerc(exec, m, n,
         alpha,
         x, incX,
         y, incY,
         A, ldA);
  } else {
    // No such implementation
    assert(false);  // XXX: Use fn which does not elide with NDEBUG
  }
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
void
geru(const execution_policy<DerivedPolicy>& exec,
    int n,
    const VX* x, int incX,
    const VY* y, int incY,
    R& result)
{
  geru(handle(derived_cast(exec)), n,
       x, incX,
       y, incY,
       &result);
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
  } else { // RowMajor: swap x & y
    geru(exec, n, m,
         alpha,
         y, incY,
         x, incX,
         A, ldA);
  }
}

} // end namespace cublas
} // end namespace blam
