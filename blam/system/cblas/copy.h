#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// scopy
template <typename DerivedPolicy>
void
copy(const execution_policy<DerivedPolicy>& /*exec*/,
     int n,
     const float* x, int incX,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_scopy");

  cblas_scopy(n, x, incX, y, incY);
}

// dcopy
template <typename DerivedPolicy>
void
copy(const execution_policy<DerivedPolicy>& /*exec*/,
     int n,
     const double* x, int incX,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_dcopy");

  cblas_dcopy(n, x, incX, y, incY);
}

// ccopy
template <typename DerivedPolicy>
void
copy(const execution_policy<DerivedPolicy>& /*exec*/,
     int n,
     const ComplexFloat* x, int incX,
     ComplexFloat *y, int incY)
{
  BLAM_DEBUG_OUT("cblas_ccopy");

  cblas_ccopy(n, reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<float*>(y), incY);
}

// zcopy
template <typename DerivedPolicy>
void
copy(const execution_policy<DerivedPolicy>& /*exec*/,
     int n,
     const ComplexDouble* x, int incX,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_zcopy");

  cblas_zcopy(n, reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<double*>(y), incY);
}

} // end namespace cblas
} // end namespace blam
