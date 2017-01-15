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
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// sdsdot
void
sdotu(int n, const float& alpha,
      const float* x, int incX,
      const float* y, int incY,
      float& result)
{
  BLAM_DEBUG_OUT("cblas_sdsdot");

  result = cblas_sdsdot(n, alpha, x, incX, y, incY);
}

// dsdot
void
dotu(int n,
     const float* x, int incX,
     const float* y, int incY,
     double& result)
{
  BLAM_DEBUG_OUT("cblas_dsdot");

  result = cblas_dsdot(n, x, incX, y, incY);
}

// sdot
void
dotu(int n,
     const float* x, int incX,
     const float* y, int incY,
     float& result)
{
  BLAM_DEBUG_OUT("cblas_sdot");

  result = cblas_sdot(n, x, incX, y, incY);
}

// ddot
void
dotu(int n,
     const double* x, int incX,
     const double* y, int incY,
     double& result)
{
  BLAM_DEBUG_OUT("cblas_ddot");

  result = cblas_ddot(n, x, incX, y, incY);
}

// cdotu_sub
void
dotu(int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat& result)
{
  BLAM_DEBUG_OUT("cblas_cdotu_sub");

  cblas_cdotu_sub(n, reinterpret_cast<const float*>(x), incX,
                  reinterpret_cast<const float*>(y), incY,
                  reinterpret_cast<float*>(&result));
}

// zdotu_sub
void
dotu(int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble& result)
{
  BLAM_DEBUG_OUT("cblas_zdotu_sub");

  cblas_zdotu_sub(n, reinterpret_cast<const double*>(x), incX,
                  reinterpret_cast<const double*>(y), incY,
                  reinterpret_cast<double*>(&result));
}

// cdotc_sub
void
dotc(int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat& result)
{
  BLAM_DEBUG_OUT("cblas_cdotc_sub");

  cblas_cdotc_sub(n, reinterpret_cast<const float*>(x), incX,
                  reinterpret_cast<const float*>(y), incY,
                  reinterpret_cast<float*>(&result));
}

// zdotc_sub
void
dotc(int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble& result)
{
  BLAM_DEBUG_OUT("cblas_zdotc_sub");

  cblas_zdotc_sub(n, reinterpret_cast<const double*>(x), incX,
                  reinterpret_cast<const double*>(y), incY,
                  reinterpret_cast<double*>(&result));
}

// blam -> cblas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
auto
dotu(const execution_policy<DerivedPolicy>& /*exec*/, int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
    -> decltype(dotu(n, x, incX, y, incY, result))
{
  return dotu(n, x, incX, y, incY, result);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
auto
dotc(const execution_policy<DerivedPolicy>& /*exec*/, int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
    -> decltype(dotc(n, x, incX, y, incY, result))
{
  return dotc(n, x, incX, y, incY, result);
}

} // end namespace cblas
} // end namespace blam
