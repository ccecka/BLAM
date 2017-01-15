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

// sscal
void
scal(int n, const float& alpha, float* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_sscal");

  cblas_sscal(n, alpha, x, incX);
}

// dscal
void
scal(int n, const double& alpha, double* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_dscal");

  cblas_dscal(n, alpha, x, incX);
}

// cscal
void
scal(int n, const ComplexFloat& alpha, ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_cscal");

  cblas_cscal(n, reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<float*>(x), incX);
}

// zscal
void
scal(int n, const ComplexDouble& alpha, ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_zscal");

  cblas_zscal(n, reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<double*>(x), incX);
}

// csscal
void
scal(int n, const float& alpha, ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_csscal");

  cblas_csscal(n, alpha, reinterpret_cast<float*>(x), incX);
}

// zdscal
void
scal(int n, const double& alpha, ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_zdscal");

  cblas_zdscal(n, alpha, reinterpret_cast<double*>(x), incX);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename VX>
auto
scal(const execution_policy<DerivedPolicy>& /*exec*/,
     int n, const Alpha& alpha,
     const VX* x, int incX)
    -> decltype(scal(n, alpha, x, incX))
{
  return scal(n, alpha, x, incX);
}

} // end namespace cblas
} // end namespace blam
