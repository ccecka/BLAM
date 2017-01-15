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
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sspr
void
spr(cublasHandle_t handle, cublasFillMode_t upLo,
    int n,
    const float* alpha,
    const float* x, int incX,
    float* A)
{
  BLAM_DEBUG_OUT("cublasSspr");

  cublasSspr(handle, upLo,
             n,
             alpha,
             x, incX,
             A);
}

// dspr
void
spr(cublasHandle_t handle, cublasFillMode_t upLo,
    int n,
    const double* alpha,
    const double* x, int incX,
    double* A)
{
  BLAM_DEBUG_OUT("cublasDspr");

  cublasDspr(handle, upLo,
             n,
             alpha,
             x, incX,
             A);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename MA>
auto
spr(const execution_policy<DerivedPolicy>& exec,
    StorageUpLo upLo,
    int n,
    const Alpha& alpha,
    const VX* x, int incX,
    MA* A)
    -> decltype(spr(handle(derived_cast(exec)), cublas_type(upLo),
                    n, &alpha,
                    x, incX,
                    A))
{
  return spr(handle(derived_cast(exec)), cublas_type(upLo),
             n, &alpha,
             x, incX,
             A);
}

// XXX TODO RowMajor -> ColMajor?

} // end namespace cublas
} // end namespace blam
