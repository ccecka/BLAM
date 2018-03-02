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

namespace blam
{
namespace cblas
{

// snrm2
inline void
nrm2(int n, const float* x, int incX, float& norm)
{
  BLAM_DEBUG_OUT("cblas_snrm2");

  norm = cblas_snrm2(n, x, incX);
}

// dnrm2
inline void
nrm2(int n, const double* x, int incX, double& norm)
{
  BLAM_DEBUG_OUT("cblas_dnrm2");

  norm = cblas_dnrm2(n, x, incX);
}

// scnrm2
inline void
nrm2(int n, const ComplexFloat* x, int incX, float& norm)
{
  BLAM_DEBUG_OUT("cblas_scnrm2");

  norm = cblas_scnrm2(n, reinterpret_cast<const float*>(x), incX);
}

// dznrm2
inline void
nrm2(int n, const ComplexDouble* x, int incX, double& norm)
{
  BLAM_DEBUG_OUT("cblas_dznrm2");

  norm = cblas_dznrm2(n, reinterpret_cast<const double*>(x), incX);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename VX, typename R>
inline auto
nrm2(const execution_policy<DerivedPolicy>& /*exec*/, int n,
     const VX* x, int incX,
     R& result)
    -> decltype(nrm2(n, x, incX, result))
{
  return nrm2(n, x, incX, result);
}

} // end namespace cblas
} // end namespace blam
