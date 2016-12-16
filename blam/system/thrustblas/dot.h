/******************************************************************************
 * Copyright (C) 2016, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016, NVIDIA CORPORATION.  All rights reserved.
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
