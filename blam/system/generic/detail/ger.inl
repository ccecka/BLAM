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
#include <blam/adl/level2/ger.h>

#if defined(BLAM_USE_DECAY)
//# include <blam/adl/level1/axpy.h>
#endif

namespace blam
{
namespace system
{
namespace generic
{

template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
gerc(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
{
#if defined(BLAM_USE_DECAY)
  // axpy
#else
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
#endif
}

template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
geru(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
{
#if defined(BLAM_USE_DECAY)
  // axpy
#else
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
#endif
}

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
ger(const ExecutionPolicy& exec,
    int m, int n,
    const Alpha& alpha,
    const VX* x, int incX,
    const VY* y, int incY,
    MA* A, int ldA)
{
  blam::adl::ger(exec,
                 ColMajor, m, n,
                 alpha,
                 x, incX,
                 y, incY,
                 A, ldA);
}

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
gerc(const ExecutionPolicy& exec,
    int m, int n,
    const Alpha& alpha,
    const VX* x, int incX,
    const VY* y, int incY,
    MA* A, int ldA)
{
  blam::adl::gerc(exec,
                  ColMajor, m, n,
                  alpha,
                  x, incX,
                  y, incY,
                  A, ldA);
}

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
geru(const ExecutionPolicy& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
{
  blam::adl::geru(exec,
                  ColMajor, m, n,
                  alpha,
                  x, incX,
                  y, incY,
                  A, ldA);
}

// sger -> sgeru
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const float& alpha,
    const float* x, int incX,
    const float* y, int incY,
    MA* A, int ldA)
{
  blam::adl::geru(exec,
                 order, m, n,
                 alpha,
                 x, incX,
                 y, incY,
                 A, ldA);
}

// dger -> dgeru
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const double& alpha,
    const double* x, int incX,
    const double* y, int incY,
    MA* A, int ldA)
{
  blam::adl::geru(exec,
                  order, m, n,
                  alpha,
                  x, incX,
                  y, incY,
                  A, ldA);
}

// cger -> cgerc
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const ComplexFloat& alpha,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    MA* A, int ldA)
{
  blam::adl::gerc(exec,
                  order, m, n,
                  alpha,
                  x, incX,
                  y, incY,
                  A, ldA);
}

// zger -> zgerc
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const ComplexDouble& alpha,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    MA* A, int ldA)
{
  blam::adl::gerc(exec,
                  order, m, n,
                  alpha,
                  x, incX,
                  y, incY,
                  A, ldA);
}

} // end namespace generic
} // end namespace system
} // end namespace blam
