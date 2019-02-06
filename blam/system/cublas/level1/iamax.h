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

#include <blam/system/cublas/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// isamax
inline cublasStatus_t
iamax(cublasHandle_t handle, int n,
      const float* x, int incX,
      int* result)
{
  BLAM_DEBUG_OUT("cublasIsamax");

  return cublasIsamax(handle, n,
                      x, incX,
                      result);
}


// idamax
inline cublasStatus_t
iamax(cublasHandle_t handle, int n,
      const double* x, int incX,
      int* result)
{
  BLAM_DEBUG_OUT("cublasIdamax");

  return cublasIdamax(handle, n,
                      x, incX,
                      result);
}


// icamax
inline cublasStatus_t
iamax(cublasHandle_t handle, int n,
      const ComplexFloat* x, int incX,
      int* result)
{
  BLAM_DEBUG_OUT("cublasIcamax");

  return cublasIcamax(handle, n,
                      reinterpret_cast<const cuFloatComplex*>(x), incX,
                      result);
}

// izamax
inline cublasStatus_t
iamax(cublasHandle_t handle, int n,
      const ComplexDouble* x, int incX,
      int* result)
{
  BLAM_DEBUG_OUT("cublasIzamax");

  return cublasIzamax(handle, n,
                      reinterpret_cast<const cuDoubleComplex*>(x), incX,
                      result);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename R>
inline auto
iamax(const execution_policy<DerivedPolicy>& exec, int n,
      const VX* x, int incX,
      R& result)
    -> decltype(iamax(handle(derived_cast(exec)), n,
                      x, incX,
                      std::declval<int*>()))   // XXX HACK for 64-bit/32-bit
{
  int result_i;
  auto r = iamax(handle(derived_cast(exec)), n,
                 x, incX,
                 &result_i);
  result = --result_i;
  return r;
}

} // end namespace cublas
} // end namespace blam
