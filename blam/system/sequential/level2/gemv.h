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

#include <blam/system/sequential/config.h>
#include <blam/system/sequential/execution_policy.h>

#include <blam/blas/level1/dot.h>

namespace blam
{
namespace seq
{

template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
inline void
gemv(const execution_policy<DerivedPolicy>& exec,
     Op trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
{
  using Ax_t = decltype(*A * *x);

  int stepA = 1;
  if (trans & Op::Trans) {
    std::swap(stepA, ldA);
    std::swap(m, n);
  }

  if (incY < 0) y -= incY*(m-1);

  if (trans & Op::Conj) { // is conjugated -> use dotc
    for (int i = 0; i < m; ++i, y += incY, A += stepA) {
      Ax_t Ax;
      blam::dotc(derived_cast(exec), n, A, ldA, x, incX, Ax);
      *y = alpha * Ax + beta * *y;
    }
  } else {                // is not conjugated -> use dotu
    for (int i = 0; i < m; ++i, y += incY, A += stepA) {
      Ax_t Ax;
      blam::dotu(derived_cast(exec), n, A, ldA, x, incX, Ax);
      *y = alpha * Ax + beta * *y;
    }
  }
}

template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
inline void
gemv(const execution_policy<DerivedPolicy>& exec,
     Layout order, Op trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
{
  if (order == RowMajor) {
    trans = Op(trans ^ Op::Trans);
    std::swap(m,n);
  }

  return gemv(exec, trans,
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

} // end namespace seq
} // end namespace blam
