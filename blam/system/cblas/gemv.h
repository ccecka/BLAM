#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// sgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, Transpose trans,
     int m, int n,
     const float& alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float& beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_sgemv");

  cblas_sgemv(cblas_order(order), cblas_transpose(trans),
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// dgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, Transpose trans,
     int m, int n,
     const double& alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double& beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_dgemv");

  cblas_dgemv(cblas_order(order), cblas_transpose(trans),
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// cgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, Transpose trans,
     int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat& beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_cgemv");

  cblas_cgemv(cblas_order(order), cblas_transpose(trans),
              m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(&beta),
              reinterpret_cast<float*>(y), incY);
}

// zgemv
template <typename DerivedPolicy>
void
gemv(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, Transpose trans,
     int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble& beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_zgemv");

  cblas_zgemv(cblas_order(order), cblas_transpose(trans),
              m, n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(&beta),
              reinterpret_cast<double*>(y), incY);
}

} // end namespace cblas
} // end namespace blam
