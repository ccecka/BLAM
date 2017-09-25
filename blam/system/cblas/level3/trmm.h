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

// strmm
void
trmm(CBLAS_LAYOUT order, CBLAS_SIDE side, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
     int m, int n,
     const float& alpha,
     const float* A, int ldA,
     float* B, int ldB)
{
  BLAM_DEBUG_OUT("cblas_strmm");

  cblas_strmm(order, side, uplo, transA, diag,
              m, n,
              alpha,
              A, ldA,
              B, ldB);
}

// dtrmm
void
trmm(CBLAS_LAYOUT order, CBLAS_SIDE side, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
     int m, int n,
     const double& alpha,
     const double* A, int ldA,
     double* B, int ldB)
{
  BLAM_DEBUG_OUT("cblas_dtrmm");

  cblas_dtrmm(order, side, uplo, transA, diag,
              m, n,
              alpha,
              A, ldA,
              B, ldB);
}

// ctrmm
void
trmm(CBLAS_LAYOUT order, CBLAS_SIDE side, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
     int m, int n,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     ComplexFloat* B, int ldB)
{
  BLAM_DEBUG_OUT("cblas_ctrmm");

  cblas_ctrmm(order, side, uplo, transA, diag,
              m, n,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<float*>(B), ldB);
}

// ztrmm
void
trmm(CBLAS_LAYOUT order, CBLAS_SIDE side, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
     int m, int n,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     ComplexDouble* B, int ldB)
{
  BLAM_DEBUG_OUT("cblas_ztrmm");

  cblas_ztrmm(order, side, uplo, transA, diag,
              m, n,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<double*>(B), ldB);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB>
auto
trmm(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Side side, Uplo uplo, Op transA, Diag diag,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     MB* B, int ldB)
    -> decltype(trmm(cblas_type(order), cblas_type(side), cblas_type(uplo),
                     cblas_type(transA), cblas_type(diag),
                     m, n,
                     alpha,
                     A, ldA,
                     B, ldB))
{
  return trmm(cblas_type(order), cblas_type(side), cblas_type(uplo),
              cblas_type(transA), cblas_type(diag),
              m, n,
              alpha,
              A, ldA,
              B, ldB);
}

} // end namespace cblas
} // end namespace blam
