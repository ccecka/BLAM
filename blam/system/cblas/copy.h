#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// scopy
void
copy(int n,
     const float* x, int incX,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_scopy");

  cblas_scopy(n, x, incX, y, incY);
}

// dcopy
void
copy(int n,
     const double* x, int incX,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_dcopy");

  cblas_dcopy(n, x, incX, y, incY);
}

// ccopy
void
copy(int n,
     const ComplexFloat* x, int incX,
     ComplexFloat *y, int incY)
{
  BLAM_DEBUG_OUT("cblas_ccopy");

  cblas_ccopy(n, reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<float*>(y), incY);
}

// zcopy
void
copy(int n,
     const ComplexDouble* x, int incX,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_zcopy");

  cblas_zcopy(n, reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<double*>(y), incY);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename VX, typename VY>
auto
copy(const execution_policy<DerivedPolicy>& /*exec*/,
     int n,
     const VX* x, int incX,
     VY* y, int incY)
    -> decltype(copy(n, x, incX, y, incY))
{
  return copy(n, x, incX, y, incY);
}

} // end namespace cblas
} // end namespace blam
