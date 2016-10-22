#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// sdsdot
template <typename DerivedPolicy>
void
sdotu(int n, const float& alpha,
      const float* x, int incX,
      const float* y, int incY,
      float* result)
{
  BLAM_DEBUG_OUT("cblas_sdsdot");

  *result = cblas_sdsdot(n, alpha, x, incX, y, incY);
}

// dsdot
template <typename DerivedPolicy>
void
dotu(const execution_policy<DerivedPolicy>& /*exec*/,
     int n,
     const float* x, int incX,
     const float* y, int incY,
     double* result)
{
  BLAM_DEBUG_OUT("cblas_dsdot");

  *result = cblas_dsdot(n, x, incX, y, incY);
}

// sdot
void
dotu(int n,
     const float* x, int incX,
     const float* y, int incY,
     float* result)
{
  BLAM_DEBUG_OUT("cblas_sdot");

  *result = cblas_sdot(n, x, incX, y, incY);
}

// ddot
void
dotu(int n,
     const double* x, int incX,
     const double* y, int incY,
     double* result)
{
  BLAM_DEBUG_OUT("cblas_ddot");

  *result = cblas_ddot(n, x, incX, y, incY);
}

// cdotc_sub
void
dotc(int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* result)
{
  BLAM_DEBUG_OUT("cblas_cdotc_sub");

  cblas_cdotc_sub(n, reinterpret_cast<const float*>(x), incX,
                  reinterpret_cast<const float*>(y), incY,
                  reinterpret_cast<float*>(result));
}

// zdotc_sub
void
dotc(int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* result)
{
  BLAM_DEBUG_OUT("cblas_zdotc_sub");

  cblas_zdotc_sub(n, reinterpret_cast<const double*>(x), incX,
                  reinterpret_cast<const double*>(y), incY,
                  reinterpret_cast<double*>(result));
}

// cdotu_sub
void
dotu(int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* result)
{
  BLAM_DEBUG_OUT("cblas_cdotu_sub");

  cblas_cdotu_sub(n, reinterpret_cast<const float*>(x), incX,
                  reinterpret_cast<const float*>(y), incY,
                  reinterpret_cast<float*>(result));
}

// zdotu_sub
void
dotu(int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* result)
{
  BLAM_DEBUG_OUT("cblas_zdotu_sub");

  cblas_zdotu_sub(n, reinterpret_cast<const double*>(x), incX,
                  reinterpret_cast<const double*>(y), incY,
                  reinterpret_cast<double*>(result));
}

// blam -> cblas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
auto
dotc(const execution_policy<DerivedPolicy>& /*exec*/, int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
    -> decltype(dotc(n,
                     x, incX,
                     y, incY,
                     &result))
{
  return dotc(n,
              x, incX,
              y, incY,
              &result);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
auto
dotu(const execution_policy<DerivedPolicy>& /*exec*/, int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
    -> decltype(dotu(n,
                     x, incX,
                     y, incY,
                     &result))
{
  return dotu(n,
              x, incX,
              y, incY,
              &result);
}

} // end namespace cblas
} // end namespace blam
