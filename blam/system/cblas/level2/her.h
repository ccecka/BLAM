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
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// cgerc
void
her(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
    int n,
    const float& alpha,
    const ComplexFloat* x, int incX,
    ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_cher");

  cblas_cher(order, upLo,
             n, alpha,
             reinterpret_cast<const float*>(x), incX,
             reinterpret_cast<float*>(A), ldA);
}

// zgerc
void
her(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
    int n,
    const double& alpha,
    const ComplexDouble* x, int incX,
    ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_zher");

  cblas_zher(order, upLo,
             n, alpha,
             reinterpret_cast<const double*>(x), incX,
             reinterpret_cast<double*>(A), ldA);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename MA>
auto
her(const execution_policy<DerivedPolicy>& /*exec*/,
    StorageOrder order, StorageUpLo upLo,
    int n,
    const Alpha& alpha,
    const VX* x, int incX,
    MA* A, int ldA)
    -> decltype(her(cblas_type(order), cblas_type(upLo),
                    n, alpha,
                    x, incX,
                    A, ldA))
{
  return her(cblas_type(order), cblas_type(upLo),
             n, alpha,
             x, incX,
             A, ldA);
}

} // end namespace cblas
} // end namespace blam
