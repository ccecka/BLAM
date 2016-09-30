#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// sger
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const float& alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_sger");

  cblas_sger(cblas_order(order),
             m, n,
             alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// dger
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const double& alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_dger");

  cblas_dger(cblas_order(order),
             m, n,
             alpha,
             x, incX,
             y, incY,
             A, ldA);
}

// cgerc
template <typename DerivedPolicy>
void
gerc(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_cgerc");

  cblas_cgerc(cblas_order(order),
              m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(y), incY,
              reinterpret_cast<float*>(A), ldA);
}

// zgerc
template <typename DerivedPolicy>
void
gerc(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_zgerc");

  cblas_zgerc(cblas_order(order),
              m, n,
              reinterpret_cast<const double* >(&alpha),
              reinterpret_cast<const double* >(x), incX,
              reinterpret_cast<const double* >(y), incY,
              reinterpret_cast<double* >(A), ldA);
}

// cgeru
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_cgeru");

  cblas_cgeru(cblas_order(order),
              m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(y), incY,
              reinterpret_cast<float*>(A), ldA);
}

// zgeru
template <typename DerivedPolicy>
void
geru(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_zgeru");

  cblas_zgeru(cblas_order(order),
              m, n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(y), incY,
              reinterpret_cast<double*>(A), ldA);
}

} // end namespace cblas
} // end namespace blam
