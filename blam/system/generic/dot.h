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
#include <blam/dot.h>

namespace blam
{
namespace adl
{

template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
generic(blam::_dot, const ExecutionPolicy& exec,
        int n,
        const VX* x, int incX,
        const VY* y, int incY,
        R& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
generic(blam::_dotc, const ExecutionPolicy& exec,
        int n,
        const VX* x, int incX,
        const VY* y, int incY,
        R& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
generic(blam::_dotu, const ExecutionPolicy& exec,
        int n,
        const VX* x, int incX,
        const VY* y, int incY,
        R& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
generic(blam::_dot, const ExecutionPolicy& exec,
        int n,
        const VX* x,
        const VY* y,
        R& result)
{
  blam::dot(exec, n, x, 1, y, 1, result);
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
generic(blam::_dotc, const ExecutionPolicy& exec,
        int n,
        const VX* x,
        const VY* y,
        R& result)
{
  blam::dotc(exec, n, x, 1, y, 1, result);
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
generic(blam::_dotu, const ExecutionPolicy& exec,
        int n,
        const VX* x,
        const VY* y,
        R& result)
{
  blam::dotu(exec, n, x, 1, y, 1, result);
}

// sdot -> sdotu
template <typename ExecutionPolicy,
          typename R>
void
generic(blam::_dot, const ExecutionPolicy& exec,
        int n,
        const float* x, int incX,
        const float* y, int incY,
        R& result)
{
  blam::dotu(exec, n, x, incX, y, incY, result);
}

// ddot -> ddotu
template <typename ExecutionPolicy,
          typename R>
void
generic(blam::_dot, const ExecutionPolicy& exec,
        int n,
        const double* x, int incX,
        const double* y, int incY,
        R& result)
{
  blam::dotu(exec, n, x, incX, y, incY, result);
}

// cdot -> cdotc
template <typename ExecutionPolicy,
          typename R>
void
generic(blam::_dot, const ExecutionPolicy& exec,
        int n,
        const ComplexFloat* x, int incX,
        const ComplexFloat* y, int incY,
        R& result)
{
  blam::dotc(exec, n, x, incX, y, incY, result);
}

// zdot -> zdotc
template <typename ExecutionPolicy,
          typename R>
void
generic(blam::_dot, const ExecutionPolicy& exec,
        int n,
        const ComplexDouble* x, int incX,
        const ComplexDouble* y, int incY,
        R& result)
{
  blam::dotc(exec, n, x, incX, y, incY, result);
}

} // end namespace adl
} // end namespace blam
