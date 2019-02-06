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

#include <blam/system/cublas/level3/trmm.h>  // RowMajor+ConjTrans

namespace blam
{
namespace cublas
{

// strmv
inline cublasStatus_t
trmv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
     int n,
     const float* A, int ldA,
     float* x, int incX)
{
  BLAM_DEBUG_OUT("cublasStrmv");

  return cublasStrmv(handle, uplo, trans, diag,
                     n,
                     A, ldA,
                     x, incX);
}

// dtrmv
inline cublasStatus_t
trmv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
     int n,
     const double* A, int ldA,
     double* x, int incX)
{
  BLAM_DEBUG_OUT("cublasDtrmv");

  return cublasDtrmv(handle, uplo, trans, diag,
                     n,
                     A, ldA,
                     x, incX);
}

// ctrmv
inline cublasStatus_t
trmv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
     int n,
     const ComplexFloat* A, int ldA,
     ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cublasCtrmv");

  return cublasCtrmv(handle, uplo, trans, diag,
                     n,
                     reinterpret_cast<const cuFloatComplex*>(A), ldA,
                     reinterpret_cast<cuFloatComplex*>(x), incX);
}

// ztrmv
inline cublasStatus_t
trmv(cublasHandle_t handle,
     cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
     int n,
     const ComplexDouble* A, int ldA,
     ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cublasZtrmv");

  return cublasZtrmv(handle, uplo, trans, diag,
                     n,
                     reinterpret_cast<const cuDoubleComplex*>(A), ldA,
                     reinterpret_cast<cuDoubleComplex*>(x), incX);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename MA, typename VX>
inline auto
trmv(const execution_policy<DerivedPolicy>& exec,
     Uplo uplo, Op trans, Diag diag,
     int n,
     const MA* A, int ldA,
     VX* x, int incX)
    -> decltype(trmv(handle(derived_cast(exec)), cublas_type(uplo),
                     cublas_type(trans), cublas_type(diag),
                     n,
                     A, ldA,
                     x, incX))
{
  return trmv(handle(derived_cast(exec)), cublas_type(uplo),
              cublas_type(trans), cublas_type(diag),
              n,
              A, ldA,
              x, incX);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename MA, typename VX>
inline auto
trmv(const execution_policy<DerivedPolicy>& exec,
     Layout order, Uplo uplo, Op trans, Diag diag,
     int n,
     const MA* A, int ldA,
     VX* x, int incX)
    -> decltype(trmv(exec, uplo, trans, diag,
                     n,
                     A, ldA,
                     x, incX))
{
  if (order == RowMajor) {
    if ((std::is_same<MA,ComplexFloat>::value || std::is_same<MA,ComplexDouble>::value)
        && trans == ConjTrans) {
      // No zero-overhead solution exists for RowMajor+Complex+ConjTrans trmv. Options are:
      // 0) Fail with return code, assert, or throw
      // 1) Decay to many dot/axpy
      // 2) Conjugate x on input and output
      // 3) Promote to trmm

      // Here, we've chosen (3), which works when incX > 0
      // (Could consider a copy for incX < 0)

      //assert(false && "No cublas::trmv for RowMajor+ConjTrans");
      //return CUBLAS_STATUS_INVALID_VALUE;

      MA alpha = 1;
      return trmm(exec, order, Left, uplo, trans, diag,
                  n, 1,
                  alpha,
                  A, ldA,
                  x, incX);
    }
    // A => A^T; A^T => A; A^H => A, swap upper <=> lower
    uplo = (uplo==Upper) ? Lower : Upper;
    trans = (trans==NoTrans ? Trans : NoTrans);
  }

  return trmv(exec, uplo, trans, diag,
              n,
              A, ldA,
              x, incX);
}

} // end namespace cublas
} // end namespace blam
