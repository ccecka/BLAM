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

// stpsv
void
tpsv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t transA, cublasDiagType_t diag,
     int n,
     const float* A,
     float* x, int incX)
{
  BLAM_DEBUG_OUT("cublasStpsv");

  cublasStpsv(handle, uplo, transA, diag,
              n, A,
              x, incX);
}

// dtpsv
void
tpsv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t transA, cublasDiagType_t diag,
     int n,
     const double* A,
     double* x, int incX)
{
  BLAM_DEBUG_OUT("cublasDtpsv");

  cublasDtpsv(handle, uplo, transA, diag,
              n, A,
              x, incX);
}

// ctpsv
void
tpsv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t transA, cublasDiagType_t diag,
     int n,
     const ComplexFloat* A,
     ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cublasCtpsv");

  cublasCtpsv(handle, uplo, transA, diag,
              n, reinterpret_cast<const cuFloatComplex*>(A),
              reinterpret_cast<cuFloatComplex*>(x), incX);
}

// ztpsv
void
tpsv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t transA, cublasDiagType_t diag,
     int n,
     const ComplexDouble* A,
     ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cublasZtpsv");

  cublasZtpsv(handle, uplo, transA, diag,
              n, reinterpret_cast<const cuDoubleComplex*>(A),
              reinterpret_cast<cuDoubleComplex*>(x), incX);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename MA, typename VX>
auto
tpsv(const execution_policy<DerivedPolicy>& exec,
     Uplo uplo, Op transA, Diag diag,
     int n,
     const MA* A,
     VX* x, int incX)
    -> decltype(tpsv(handle(derived_cast(exec)), cublas_type(uplo),
                     cublas_type(transA), cublas_type(diag),
                     n, A,
                     x, incX))
{
  return tpsv(handle(derived_cast(exec)), cublas_type(uplo),
              cublas_type(transA), cublas_type(diag),
              n, A,
              x, incX);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename MA, typename VX>
auto
tpsv(const execution_policy<DerivedPolicy>& exec,
     Layout order, Uplo uplo, Op transA, Diag diag,
     int n,
     const MA* A,
     VX* x, int incX)
    -> decltype(tpsv(exec, uplo, transA, diag,
                     n, A,
                     x, incX))
{
  if (order == RowMajor) {
    transA = Op(transA ^ Trans);
    uplo = (uplo==Upper) ? Lower : Upper;
  }

  return tpsv(exec, uplo, transA, diag,
              n, A,
              x, incX);
}

} // end namespace cublas
} // end namespace blam
