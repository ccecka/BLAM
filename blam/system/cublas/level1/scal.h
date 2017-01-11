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

// sscal
void
scal(cublasHandle_t handle, int n, const float* alpha,
     float* x, int incX)
{
  BLAM_DEBUG_OUT("cublasSscal");

  cublasSscal(handle, n, alpha, x, incX);
}

// dscal
void
scal(cublasHandle_t handle, int n, const double* alpha,
     double* x, int incX)
{
  BLAM_DEBUG_OUT("cublasDscal");

  cublasDscal(handle, n, alpha, x, incX);
}

// csscal
void
scal(cublasHandle_t handle, int n, const float* alpha,
     ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cublasCsscal");

  cublasCsscal(handle, n, alpha, reinterpret_cast<cuFloatComplex*>(x), incX);
}

// csscal
void
scal(cublasHandle_t handle, int n, const ComplexFloat* alpha,
     ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cublasCscal");

  cublasCscal(handle, n,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<cuFloatComplex*>(x), incX);
}

// zdscal
void
scal(cublasHandle_t handle, int n, const double* alpha,
     ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cublasZdscal");

  cublasZdscal(handle, n,
               alpha,
               reinterpret_cast<cuDoubleComplex*>(x), incX);
}

// zscal
void
scal(cublasHandle_t handle, int n, const ComplexDouble* alpha,
     ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cublasZdscal");

  cublasZscal(handle, n,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<cuDoubleComplex*>(x), incX);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename VX>
auto
scal(const execution_policy<DerivedPolicy>& exec,
     int n, const Alpha& alpha,
     const VX* x, int incX)
    -> decltype(scal(handle(derived_cast(exec)), n, alpha, x, incX))
{
  return scal(handle(derived_cast(exec)), n, alpha, x, incX);
}

} // end namespace cublas
} // end namespace blam
