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

// sdotu
inline cublasStatus_t
dotu(cublasHandle_t handle, int n,
     const float* x, int incX,
     const float* y, int incY,
     float* result)
{
  BLAM_DEBUG_OUT("cublasSdot");

  return cublasSdot(handle, n,
                    x, incX,
                    y, incY,
                    result);
}

// ddotu
inline cublasStatus_t
dotu(cublasHandle_t handle, int n,
     const double* x, int incX,
     const double* y, int incY,
     double* result)
{
  BLAM_DEBUG_OUT("cublasDdot");

  return cublasDdot(handle, n,
                    x, incX,
                    y, incY,
                    result);
}

// cdotu
inline cublasStatus_t
dotu(cublasHandle_t handle, int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* result)
{
  BLAM_DEBUG_OUT("cublasCdotu");

  return cublasCdotu(handle, n,
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<const cuFloatComplex*>(y), incY,
                     reinterpret_cast<cuFloatComplex*>(result));
}

// zdotu
inline cublasStatus_t
dotu(cublasHandle_t handle, int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* result)
{
  BLAM_DEBUG_OUT("cublasZdotu");

  return cublasZdotu(handle, n,
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<const cuDoubleComplex*>(y), incY,
                     reinterpret_cast<cuDoubleComplex*>(result));
}

// sdotc
inline cublasStatus_t
dotc(cublasHandle_t handle, int n,
     const float* x, int incX,
     const float* y, int incY,
     float* result)
{
  return dotu(handle, n,
              x, incX,
              y, incY,
              result);
}

// ddotc
inline cublasStatus_t
dotc(cublasHandle_t handle, int n,
     const double* x, int incX,
     const double* y, int incY,
     double* result)
{
  return dotu(handle, n,
              x, incX,
              y, incY,
              result);
}

// cdotc
inline cublasStatus_t
dotc(cublasHandle_t handle, int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* result)
{
  BLAM_DEBUG_OUT("cublasCdotc");

  return cublasCdotc(handle, n,
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<const cuFloatComplex*>(y), incY,
                     reinterpret_cast<cuFloatComplex*>(result));
}

// zdotc
inline cublasStatus_t
dotc(cublasHandle_t handle, int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* result)
{
  BLAM_DEBUG_OUT("cublasZdotc");

  return cublasZdotc(handle, n,
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<const cuDoubleComplex*>(y), incY,
                     reinterpret_cast<cuDoubleComplex*>(result));
}

// sdot
inline cublasStatus_t
dot(cublasHandle_t handle, int n,
    const float* x, int incX,
    const float* y, int incY,
    float* result)
{
  return dotu(handle, n,
              x, incX,
              y, incY,
              result);
}

// ddot
inline cublasStatus_t
dot(cublasHandle_t handle, int n,
    const double* x, int incX,
    const double* y, int incY,
    double* result)
{
  return dotu(handle, n,
              x, incX,
              y, incY,
              result);
}

// cdot
inline cublasStatus_t
dot(cublasHandle_t handle, int n,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    ComplexFloat* result)
{
  return dotc(handle, n,
              x, incX,
              y, incY,
              result);
}

// zdot
inline cublasStatus_t
dot(cublasHandle_t handle, int n,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    ComplexDouble* result)
{
  return dotc(handle, n,
              x, incX,
              y, incY,
              result);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
inline auto
dotu(const execution_policy<DerivedPolicy>& exec, int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
    -> decltype(dotu(handle(derived_cast(exec)), n,
                     x, incX,
                     y, incY,
                     &result))
{
  return dotu(handle(derived_cast(exec)), n,
              x, incX,
              y, incY,
              &result);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
inline auto
dotc(const execution_policy<DerivedPolicy>& exec, int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
    -> decltype(dotc(handle(derived_cast(exec)), n,
                     x, incX,
                     y, incY,
                     &result))
{
  return dotc(handle(derived_cast(exec)), n,
              x, incX,
              y, incY,
              &result);
}

} // end namespace cublas
} // end namespace blam
