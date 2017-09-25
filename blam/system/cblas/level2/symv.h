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

// ssymv
void
symv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     int n, const float& alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float& beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_ssymv");

  cblas_ssymv(order, uplo,
              n, alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// dsymv
void
symv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     int n, const double& alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double& beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_dsymv");

  cblas_dsymv(order, uplo,
              n, alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
symv(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Uplo uplo,
     int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(symv(cblas_type(order), cblas_type(uplo),
                     n, alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY))
{
  return symv(cblas_type(order), cblas_type(uplo),
              n, alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

} // end namespace cblas
} // end namespace blam
