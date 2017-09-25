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

// csbmv
void
sbmv(cublasHandle_t handle, cublasFillMode_t uplo,
     int n, int k,
     const float* alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float* beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasSsbmv");

  cublasSsbmv(handle, uplo,
              n, k,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// zsbmv
void
sbmv(cublasHandle_t handle, cublasFillMode_t uplo,
     int n, int k,
     const double* alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double* beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDsbmv");

  cublasDsbmv(handle, uplo,
              n, k,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
sbmv(const execution_policy<DerivedPolicy>& exec,
     Uplo uplo,
     int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(sbmv(handle(derived_cast(exec)), cublas_type(uplo),
                     n, k,
                     &alpha,
                     A, ldA,
                     x, incX,
                     &beta,
                     y, incY))
{
  return sbmv(handle(derived_cast(exec)), cublas_type(uplo),
              n, k,
              &alpha,
              A, ldA,
              x, incX,
              &beta,
              y, incY);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
sbmv(const execution_policy<DerivedPolicy>& exec,
     Layout order, Uplo uplo,
     int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(sbmv(exec, uplo,
                     n, k,
                     alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY))
{
  if (order == RowMajor) {
    uplo = (uplo==Upper) ? Lower : Upper;
  }

  return sbmv(exec, uplo,
              n, k,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

} // end namespace cublas
} // end namespace blam
