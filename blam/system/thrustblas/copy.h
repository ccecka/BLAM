#pragma once

#include <blam/detail/config.h>
#include <blam/system/thrustblas/execution_policy.h>

#include <blam/system/thrustblas/detail/strided_range.h>
#include <thrust/copy.h>

namespace thrust
{

// copy
template <typename DerivedPolicy,
          typename T, typename U>
void
copy(const execution_policy<DerivedPolicy>& exec,
     int n,
     const T* x, int incX,
     U* y, int incY)
{
  BLAM_DEBUG_OUT("thrust copy");

  if (incX == 1 && incY == 1) {
    thrust::copy_n(exec, x, n, y);
  } else if (incX == 1) {
    auto yi = blam::make_strided_iterator(y, incY);
    thrust::copy_n(exec, x, n, yi);
  } else if (incY == 1) {
    auto xi = blam::make_strided_iterator(x, incX);
    thrust::copy_n(exec, xi, n, y);
  } else {
    auto xi = blam::make_strided_iterator(x, incY);
    auto yi = blam::make_strided_iterator(y, incY);
    thrust::copy_n(exec, xi, n, yi);
  }
}

} // end namespace thrust
