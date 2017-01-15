/******************************************************************************
 * Copyright (C) 2016-2017, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#pragma once

#include <blam/detail/config.h>
#include <blam/system/thrustblas/execution_policy.h>

#include <blam/system/thrustblas/detail/strided_range.h>
#include <thrust/copy.h>

namespace thrust
{

namespace detail
{

template <typename Alpha, typename X, typename Y>
struct axpy
{
  Alpha a;

  __host__ __device__
  auto operator()(const X& x, const Y& y) const -> decltype(a*x+y) {
    return a*x+y;
  }
};

} // end namespace detail

// axpy
template <typename DerivedPolicy,
          typename Alpha, typename VX, typename VY>
void
axpy(const execution_policy<DerivedPolicy>& exec,
     int n,
     const Alpha& alpha,
     const VX* x, int incX,
     VY* y, int incY)
{
  BLAM_DEBUG_OUT("thrust copy");

  using axpy = detail::axpy<Alpha, VX, VY>;

  if (incX == 1 && incY == 1) {
    thrust::transform(exec, x, x+n, y, y, axpy{alpha});
  } else if (incX == 1) {
    auto yi = blam::make_strided_range(y, incY);
    thrust::transform(exec, x, x+n, yi, yi, axpy{alpha});
  } else if (incY == 1) {
    auto xi = blam::make_strided_iterator(x, incX);
    thrust::transform(exec, xi, xi+n, y, y, axpy{alpha});
  } else {
    auto xi = blam::make_strided_iterator(x, incY);
    auto yi = blam::make_strided_iterator(y, incY);
    thrust::transform(exec, xi, xi+n, yi, yi, axpy{alpha});
  }
}

} // end namespace thrust
