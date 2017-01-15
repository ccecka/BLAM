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

// scopy
void
asum(cublasHandle_t handle, int n,
     const float* x, int incX,
     float& result)
{
  BLAM_DEBUG_OUT("cublasSasum");

  cublasSasum(handle, n,
              x, incX,
              &result);
}


// dasum
void
asum(cublasHandle_t handle, int n,
     const double* x, int incX,
     double& result)
{
  BLAM_DEBUG_OUT("cublasDasum");

  cublasDasum(handle, n,
              x, incX,
              &result);
}


// casum
void
asum(cublasHandle_t handle, int n,
     const ComplexFloat* x, int incX,
     float& result)
{
  BLAM_DEBUG_OUT("cublasCasum");

  cublasScasum(handle, n,
               reinterpret_cast<const cuFloatComplex*>(x), incX,
               &result);
}

// zasum
void
asum(cublasHandle_t handle, int n,
     const ComplexDouble* x, int incX,
     double& result)
{
  BLAM_DEBUG_OUT("cublasDzasum");

  cublasDzasum(handle, n,
               reinterpret_cast<const cuDoubleComplex*>(x), incX,
               &result);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename R>
auto
asum(const execution_policy<DerivedPolicy>& exec,
     int n,
     const VX* x, int incX,
     R& result)
    -> decltype(asum(handle(derived_cast(exec)), n, x, incX, result))
{
  return asum(handle(derived_cast(exec)), n, x, incX, result);
}

} // end namespace cublas
} // end namespace blam
