#pragma once

#include <blam/detail/config.h>
#include <blam/system/thrustblas/execution_policy.h>

#include <blam/system/thrustblas/detail/strided_range.h>
#include <blam/system/thrustblas/detail/complex.h>
#include <thrust/inner_product.h>

namespace thrust
{

// dotc
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
void
dotc(const execution_policy<DerivedPolicy>& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
{
  BLAM_DEBUG_OUT("thrust dotc");

  auto yc = thrust::make_transform_iterator(y, blam::conj_fn<VY>{});

  if (incX == 1 && incY == 1) {
    result = thrust::inner_product(exec, x, x+n, yc, R(0));
  } else if (incX == 1) {
    auto xs = blam::make_strided_iterator(x, incX);
    result = thrust::inner_product(exec, xs, xs+n, yc, R(0));
  } else if (incY == 1) {
    auto ycs = blam::make_strided_iterator(yc, incY);
    result = thrust::inner_product(exec, x, x+n, ycs, R(0));
  } else {
    auto xr = blam::make_strided_range(x, x+n*incX, incX);
    auto ycs = blam::make_strided_iterator(yc, incY);
    result = thrust::inner_product(exec, xr.begin(), xr.end(), ycs, R(0));
  }
}

// dotu
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
void
dotu(const execution_policy<DerivedPolicy>& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
{
  BLAM_DEBUG_OUT("thrust dotu");

  if (incX == 1 && incY == 1) {
    result = thrust::inner_product(exec, x, x+n, y, R(0));
  } else if (incX == 1) {
    auto ys = blam::make_strided_iterator(y, incY);
    result = thrust::inner_product(exec, x, x+n, ys, R(0));
  } else if (incY == 1) {
    auto xs = blam::make_strided_iterator(x, incX);
    result = thrust::inner_product(exec, y, y+n, xs, R(0));
  } else {
    auto xr = blam::make_strided_range(x, x+n*incX, incX);
    auto ys = blam::make_strided_iterator(y, incY);
    result = thrust::inner_product(exec, xr.begin(), xr.end(), ys, R(0));
  }
}

} // end namespace thrust
