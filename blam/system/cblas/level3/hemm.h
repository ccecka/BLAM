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

#include <blam/system/cblas/level3/symm.h>  // Real-valued hemm

namespace blam
{
namespace cblas
{

// shemm
inline void
hemm(CBLAS_LAYOUT order,
     CBLAS_SIDE side, CBLAS_UPLO uplo,
     int m, int n,
     const float& alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float& beta,
     float* C, int ldC)
{
  return symm(order, side, uplo,
              m, n,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// dhemm
inline void
hemm(CBLAS_LAYOUT order,
     CBLAS_SIDE side, CBLAS_UPLO uplo,
     int m, int n,
     const double& alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     const double& beta,
     double* C, int ldC)
{
  return symm(order, side, uplo,
              m, n,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// chemm
inline void
hemm(CBLAS_LAYOUT order,
     CBLAS_SIDE side, CBLAS_UPLO uplo,
     int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     const ComplexFloat& beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_chemm");

  cblas_chemm(order, side, uplo,
              m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<const float*>(B), ldB,
              reinterpret_cast<const float*>(&beta),
              reinterpret_cast<float*>(C), ldC);
}

// zhemm
inline void
hemm(CBLAS_LAYOUT order,
     CBLAS_SIDE side, CBLAS_UPLO uplo,
     int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     const ComplexDouble& beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_zhemm");

  cblas_zhemm(order, side, uplo,
              m, n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<const double*>(B), ldB,
              reinterpret_cast<const double*>(&beta),
              reinterpret_cast<double*>(C), ldC);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
inline auto
hemm(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Side side, Uplo uplo,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
    -> decltype(hemm(cblas_type(order), cblas_type(side), cblas_type(uplo),
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB,
                     beta,
                     C, ldC))
{
  return hemm(cblas_type(order), cblas_type(side), cblas_type(uplo),
              m, n,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

} // end namespace cblas
} // end namespace blam
