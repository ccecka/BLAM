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

// chpr
inline cublasStatus_t
hpr(cublasHandle_t handle, cublasFillMode_t uplo,
    int n,
    const float* alpha,
    const ComplexFloat* x, int incX,
    ComplexFloat* A)
{
  BLAM_DEBUG_OUT("cublasChpr");

  return cublasChpr(handle, uplo,
                    n,
                    alpha,
                    reinterpret_cast<const cuFloatComplex*>(x), incX,
                    reinterpret_cast<cuFloatComplex*>(A));
}

// zhpr
inline cublasStatus_t
hpr(cublasHandle_t handle, cublasFillMode_t uplo,
    int n,
    const double* alpha,
    const ComplexDouble* x, int incX,
    ComplexDouble* A)
{
  BLAM_DEBUG_OUT("cublasZhpr");

  return cublasZhpr(handle, uplo,
                    n,
                    alpha,
                    reinterpret_cast<const cuDoubleComplex*>(x), incX,
                    reinterpret_cast<cuDoubleComplex*>(A));
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename MA>
inline auto
hpr(const execution_policy<DerivedPolicy>& exec,
    Uplo uplo,
    int n,
    const Alpha& alpha,
    const VX* x, int incX,
    MA* A)
    -> decltype(hpr(handle(derived_cast(exec)), cublas_type(uplo),
                    n, &alpha,
                    x, incX,
                    A))
{
  return hpr(handle(derived_cast(exec)), cublas_type(uplo),
             n, &alpha,
             x, incX,
             A);
}

// XXX TODO RowMajor -> ColMajor?

} // end namespace cublas
} // end namespace blam
