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
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// cspr2
void
spr2(cublasHandle_t handle, cublasFillMode_t upLo,
     int n,
     const float* alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A)
{
  BLAM_DEBUG_OUT("cublasSspr2");

  cublasSspr2(handle, upLo,
              n,
              alpha,
              x, incX,
              y, incY,
              A);
}

// zspr2
void
spr2(cublasHandle_t handle, cublasFillMode_t upLo,
     int n,
     const double* alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A)
{
  BLAM_DEBUG_OUT("cublasDspr2");

  cublasDspr2(handle, upLo,
              n,
              alpha,
              x, incX,
              y, incY,
              A);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
spr2(const execution_policy<DerivedPolicy>& exec,
     StorageUpLo upLo,
     int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A)
    -> decltype(spr2(handle(derived_cast(exec)), cublas_type(upLo),
                     n, &alpha,
                     x, incX,
                     y, incY,
                     A))
{
  return spr2(handle(derived_cast(exec)), cublas_type(upLo),
              n, &alpha,
              x, incX,
              y, incY,
              A);
}

// XXX TODO RowMajor -> ColMajor?

} // end namespace cublas
} // end namespace blam
