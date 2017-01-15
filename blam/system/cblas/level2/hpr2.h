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

// chpr2
void
hpr2(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
     int n,
     const float& alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A)
{
  BLAM_DEBUG_OUT("cblas_chpr2");

  cblas_chpr2(order, upLo,
              n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(y), incY,
              reinterpret_cast<float*>(A));
}

// zhpr2
void
hpr2(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
     int n,
     const double& alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A)
{
  BLAM_DEBUG_OUT("cblas_zhpr2");

  cblas_zhpr2(order, upLo,
              n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(y), incY,
              reinterpret_cast<double*>(A));
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
hpr2(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, StorageUpLo upLo,
     int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A)
    -> decltype(hpr2(cblas_type(order), cblas_type(upLo),
                     n, alpha,
                     x, incX,
                     y, incY,
                     A))
{
  return hpr2(cblas_type(order), cblas_type(upLo),
              n, alpha,
              x, incX,
              y, incY,
              A);
}


} // end namespace cblas
} // end namespace blam
