#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sger
void
geru(cublasHandle_t handle,
     int m, int n,
     const float* alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasSger");

  cublasSger(handle,
             m, n,
             alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// dger
void
geru(cublasHandle_t handle,
     int m, int n,
     const double* alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasDger");

  cublasDger(handle,
             m, n,
             alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// cgerc
void
gerc(cublasHandle_t handle,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCgerc");

  cublasCgerc(handle,
              m, n,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(y), incY,
              reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zgerc
void
gerc(cublasHandle_t handle,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZgerc");

  cublasZgerc(handle,
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(y), incY,
              reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

// cgeru
void
geru(cublasHandle_t handle,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCgeru");

  cublasCgeru(handle,
              m, n,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(y), incY,
              reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zgeru
void
geru(cublasHandle_t handle,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZgeru");

  cublasZgeru(handle,
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(y), incY,
              reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
geru(const execution_policy<DerivedPolicy>& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(geru(handle(derived_cast(exec)),
                     m, n, &alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return geru(handle(derived_cast(exec)),
              m, n, &alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
geru(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(geru(exec, m, n,
                     alpha,
                     x, incX,
                     y, incY,
                     A, ldA),
                geru(exec, n, m,
                     alpha,
                     y, incY,
                     x, incX,
                     A, ldA))
{
  if (order == ColMajor) {
    return geru(exec, m, n,
                alpha,
                x, incX,
                y, incY,
                A, ldA);
  } else { // RowMajor: swap x & y
    return geru(exec, n, m,
                alpha,
                y, incY,
                x, incX,
                A, ldA);
  }
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
gerc(const execution_policy<DerivedPolicy>& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(gerc(handle(derived_cast(exec)),
                     m, n, &alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return gerc(handle(derived_cast(exec)),
              m, n, &alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
gerc(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(gerc(exec, m, n,
                     alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  if (order == ColMajor || (m == n && x == y && incX == incY)) {
    return gerc(exec, m, n,
                alpha,
                x, incX,
                y, incY,
                A, ldA);
  } else {
    // No such implementation
    assert(false);  // XXX: Use fn which does not elide with NDEBUG
  }
}

} // end namespace cublas
} // end namespace blam
