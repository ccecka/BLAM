#pragma once

#include <blam/detail/config.h>
#include <blam/system/thrustblas/execution_policy.h>

#include <blam/system/thrustblas/detail/strided_range.h>
#include <blam/system/thrustblas/detail/complex.h>
#include <thrust/inner_product.h>

namespace thrust
{

// dot
template <typename DerivedPolicy,
          typename T, typename U>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const T* x, int incX,
    const T* y, int incY,
    U& result)
{
  BLAM_DEBUG_OUT("thrust dot");

  auto xc = thrust::make_transform_iterator(x, blam::conj_fn<T>{});

  if (incX == 1 && incY == 1) {
    result = thrust::inner_product(exec, y, y+n, xc, U(0));
  } else if (incX == 1) {
    auto yi = blam::make_strided_iterator(y, incY);
    result = thrust::inner_product(exec, xc, xc+n, yi, U(0));
  } else if (incY == 1) {
    auto xi = blam::make_strided_iterator(xc, incX);
    result = thrust::inner_product(exec, y, y+n, xi, U(0));
  } else {
    auto yr = blam::make_strided_range(y, y+n*incY, incY);
    auto xi = blam::make_strided_iterator(xc, incY);
    result = thrust::inner_product(exec, yr.begin(), yr.end(), xi, U(0));
  }
}

// dotu
template <typename DerivedPolicy,
          typename T, typename U>
void
dotu(const execution_policy<DerivedPolicy>& exec,
     int n,
     const T* x, int incX,
     const T* y, int incY,
     U& result)
{
  BLAM_DEBUG_OUT("thrust dotu");

  if (incX == 1 && incY == 1) {
    result = thrust::inner_product(exec, x, x+n, y, U(0));
  } else if (incX == 1) {
    auto yi = blam::make_strided_iterator(y, incY);
    result = thrust::inner_product(exec, x, x+n, yi, U(0));
  } else if (incY == 1) {
    auto xi = blam::make_strided_iterator(x, incX);
    result = thrust::inner_product(exec, y, y+n, xi, U(0));
  } else {
    auto xr = blam::make_strided_range(x, x+n*incX, incX);
    auto yi = blam::make_strided_iterator(y, incY);
    result = thrust::inner_product(exec, xr.begin(), xr.end(), yi, U(0));
  }
}

} // end namespace thrust
