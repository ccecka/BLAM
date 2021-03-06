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

// sgbmv
inline void
gbmv(CBLAS_LAYOUT order, CBLAS_TRANSPOSE trans,
     int m, int n, int kl, int ku,
     const float& alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float& beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_sgbmv");

  cblas_sgbmv(order, trans,
              m, n, kl, ku,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// dgbmv
inline void
gbmv(CBLAS_LAYOUT order, CBLAS_TRANSPOSE trans,
     int m, int n, int kl, int ku,
     const double& alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double& beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_dgbmv");

  cblas_dgbmv(order, trans,
              m, n, kl, ku,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// cgbmv
inline void
gbmv(CBLAS_LAYOUT order, CBLAS_TRANSPOSE trans,
     int m, int n, int kl, int ku,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat& beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_cgbmv");

  cblas_cgbmv(order, trans,
              m, n, kl, ku,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(&beta),
              reinterpret_cast<float*>(y), incY);
}

// zgbmv
inline void
gbmv(CBLAS_LAYOUT order, CBLAS_TRANSPOSE trans,
     int m, int n, int kl, int ku,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble& beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cblas_zgbmv");

  cblas_zgbmv(order, trans,
              m, n, kl, ku,
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
gbmv(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Op trans,
     int m, int n, int kl, int ku,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(gbmv(cblas_type(order), cblas_type(trans),
                     m, n, kl, ku,
                     alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY))
{
  if (trans != Conj) {
    return gbmv(cblas_type(order), cblas_type(trans),
                m, n, kl, ku,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
  } else {
    order = (order==RowMajor) ? ColMajor : RowMajor;
    return gbmv(cblas_type(order), cblas_type(ConjTrans),
                n, m, ku, kl,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
  }
}

} // end namespace cblas
} // end namespace blam
