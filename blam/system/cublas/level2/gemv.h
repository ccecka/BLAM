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

// sgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const float* alpha,
     const float* A, int ldA,
     const float* x, int incX,
     const float* beta,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasSgemv");

  cublasSgemv(handle, trans,
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// dgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const double* alpha,
     const double* A, int ldA,
     const double* x, int incX,
     const double* beta,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDgemv");

  cublasDgemv(handle, trans,
              m, n,
              alpha,
              A, ldA,
              x, incX,
              beta,
              y, incY);
}

// cgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* x, int incX,
     const ComplexFloat* beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCgemv");

  cublasCgemv(handle, trans,
              m, n,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<const cuFloatComplex*>(A), ldA,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(beta),
              reinterpret_cast<cuFloatComplex*>(y), incY);
}

// zgemv
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* x, int incX,
     const ComplexDouble* beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  cublasZgemv(handle, trans,
              m, n,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<const cuDoubleComplex*>(A), ldA,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(beta),
              reinterpret_cast<cuDoubleComplex*>(y), incY);
}

// csgemv   XXX: Move to general
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const float* alpha,
     const ComplexFloat* A, int ldA,
     const float* x, int incX,
     const float* beta,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  assert(incY == 1);

  cublasSgemv(handle, trans,
              2*m, n,
              reinterpret_cast<const float*>(alpha),
              reinterpret_cast<const float*>(A), 2*ldA,
              reinterpret_cast<const float*>(x), incX,
              reinterpret_cast<const float*>(beta),
              reinterpret_cast<float*>(y), incY);
}

// zdgemv   XXX: Move to general
void
gemv(cublasHandle_t handle, cublasOperation_t trans,
     int m, int n,
     const double* alpha,
     const ComplexDouble* A, int ldA,
     const double* x, int incX,
     const double* beta,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZgemv");

  assert(incY == 1);

  cublasDgemv(handle, trans,
              2*m, n,
              reinterpret_cast<const double*>(alpha),
              reinterpret_cast<const double*>(A), 2*ldA,
              reinterpret_cast<const double*>(x), incX,
              reinterpret_cast<const double*>(beta),
              reinterpret_cast<double*>(y), incY);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
gemv(const execution_policy<DerivedPolicy>& exec,
     Op trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(gemv(handle(derived_cast(exec)), cublas_type(trans),
                     m, n,
                     &alpha,
                     A, ldA,
                     x, incX,
                     &beta,
                     y, incY))
{
  return gemv(handle(derived_cast(exec)), cublas_type(trans),
              m, n,
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
gemv(const execution_policy<DerivedPolicy>& exec,
     Layout order, Op trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
    -> decltype(gemv(exec, trans,
                     m, n,
                     alpha,
                     A, ldA,
                     x, incX,
                     beta,
                     y, incY))
{
  if (order == ColMajor) {
    return gemv(exec, trans,
                m, n,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
  } else { // RowMajor: transpose A
    return gemv(exec, Op(trans ^ Trans),
                n, m,
                alpha,
                A, ldA,
                x, incX,
                beta,
                y, incY);
  }
}

} // end namespace cublas
} // end namespace blam
