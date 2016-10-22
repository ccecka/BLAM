#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// sger
void
geru(const CBLAS_LAYOUT order, int m, int n,
     const float& alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_sger");

  cblas_sger(order, m, n,
             alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// dger
void
geru(const CBLAS_LAYOUT order, int m, int n,
     const double& alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_dger");

  cblas_dger(order, m, n,
             alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// cgerc
void
gerc(const CBLAS_LAYOUT order, int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_cgerc");

  cblas_cgerc(order, m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(y), incY,
              reinterpret_cast<float*>(A), ldA);
}

// zgerc
void
gerc(const CBLAS_LAYOUT order, int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_zgerc");

  cblas_zgerc(order, m, n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(y), incY,
              reinterpret_cast<double*>(A), ldA);
}

// cgeru
void
geru(const CBLAS_LAYOUT order, int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_cgeru");

  cblas_cgeru(order, m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(y), incY,
              reinterpret_cast<float*>(A), ldA);
}

// zgeru
void
geru(const CBLAS_LAYOUT order, int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_zgeru");

  cblas_zgeru(order, m, n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(y), incY,
              reinterpret_cast<double*>(A), ldA);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
geru(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(geru(cblas_order(order),
                     m, n, alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return geru(cblas_order(order),
              m, n, alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
gerc(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(gerc(cblas_order(order),
                     m, n, alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return gerc(cblas_order(order),
              m, n, alpha,
              x, incX,
              y, incY,
              A, ldA);
}

} // end namespace cblas
} // end namespace blam
