/******************************************************************************
 * Copyright (C) 2016-2019, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <blam/system/cblas/config.h>
#include <blam/system/cblas/execution_policy.h>

#include <blam/system/cblas/level2/syr.h>  // Real-valued her

namespace blam
{
namespace cblas
{

// sher
inline void
her(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
    int n,
    const float& alpha,
    const float* x, int incX,
    float* A, int ldA)
{
  return syr(order, uplo,
             n,
             alpha,
             x, incX,
             A, ldA);
}

// dher
inline void
her(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
    int n,
    const double& alpha,
    const double* x, int incX,
    double* A, int ldA)
{
  return syr(order, uplo,
             n,
             alpha,
             x, incX,
             A, ldA);
}

// cher
inline void
her(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
    int n,
    const float& alpha,
    const ComplexFloat* x, int incX,
    ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_cher");

  cblas_cher(order, uplo,
             n, alpha,
             reinterpret_cast<const float*>(x), incX,
             reinterpret_cast<float*>(A), ldA);
}

// zher
inline void
her(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
    int n,
    const double& alpha,
    const ComplexDouble* x, int incX,
    ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cblas_zher");

  cblas_zher(order, uplo,
             n, alpha,
             reinterpret_cast<const double*>(x), incX,
             reinterpret_cast<double*>(A), ldA);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename MA>
inline auto
her(const execution_policy<DerivedPolicy>& /*exec*/,
    Layout order, Uplo uplo,
    int n,
    const Alpha& alpha,
    const VX* x, int incX,
    MA* A, int ldA)
    -> decltype(her(cblas_type(order), cblas_type(uplo),
                    n, alpha,
                    x, incX,
                    A, ldA))
{
  return her(cblas_type(order), cblas_type(uplo),
             n, alpha,
             x, incX,
             A, ldA);
}

} // end namespace cblas
} // end namespace blam
