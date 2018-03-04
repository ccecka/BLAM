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

#include <blam/system/cblas/config.h>
#include <blam/system/cblas/execution_policy.h>

#include <blam/system/cblas/level2/symv.h>  // Real-valued hemv

namespace blam
{
namespace cblas
{

// shemv
inline void
hemv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     int n,
     const float& alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float& beta,
     float* y, int incY)
{
  return symv(order, uplo,
              n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// dhemv
inline void
hemv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     int n,
     const double& alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double& beta,
     double* y, int incY)
{
  return symv(order, uplo,
              n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// chemv
inline void
hemv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     int n,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat& beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_chemv");

  cblas_chemv(order, uplo,
              n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(&beta),
              reinterpret_cast<float*>(y), incY);
}

// zhemv
inline void
hemv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     int n,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble& beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_zhemv");

  cblas_zhemv(order, uplo,
              n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(&beta),
              reinterpret_cast<double*>(y), incY);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
inline auto
hemv(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Uplo uplo,
     int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(hemv(cblas_type(order), cblas_type(uplo),
                     n,
                     alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY))
{
  return hemv(cblas_type(order), cblas_type(uplo),
              n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

} // end namespace cblas
} // end namespace blam
