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

namespace blam
{
namespace cblas
{

// ssyrk
inline void
syrk(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans,
     int n, int k,
     const float& alpha,
     const float* A, int ldA,
     const float& beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_ssyrk");

  cblas_ssyrk(order, uplo, trans,
              n, k,
              alpha,
              A, ldA,
              beta,
              C, ldC);
}

// dsyrk
inline void
syrk(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans,
     int n, int k,
     const double& alpha,
     const double* A, int ldA,
     const double& beta,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_dsyrk");

  cblas_dsyrk(order, uplo, trans,
              n, k,
              alpha,
              A, ldA,
              beta,
              C, ldC);
}

// csyrk
inline void
syrk(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans,
     int n, int k,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat& beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_csyrk");

  cblas_csyrk(order, uplo, trans,
              n, k,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<const float*>(&beta),
              reinterpret_cast<float*>(C), ldC);
}

// zsyrk
inline void
syrk(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans,
     int n, int k,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble& beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_zsyrk");

  cblas_zsyrk(order, uplo, trans,
              n, k,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<const double*>(&beta),
              reinterpret_cast<double*>(C), ldC);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename MA,
          typename Beta, typename MC>
inline auto
syrk(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Uplo uplo, Op trans,
     int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const Beta& beta,
     MC* C, int ldC)
    -> decltype(syrk(cblas_type(order), cblas_type(uplo), cblas_type(trans),
                     n, k,
                     alpha,
                     A, ldA,
                     beta,
                     C, ldC))
{
  return syrk(cblas_type(order), cblas_type(uplo), cblas_type(trans),
              n, k,
              alpha,
              A, ldA,
              beta,
              C, ldC);
}

} // end namespace cblas
} // end namespace blam
