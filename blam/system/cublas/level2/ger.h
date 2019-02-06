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

#include <blam/system/cublas/level3/gemm.h>  // RowMajor gerc

namespace blam
{
namespace cublas
{

// sgeru
inline cublasStatus_t
geru(cublasHandle_t handle,
     int m, int n,
     const float* alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasSger");

  return cublasSger(handle,
                    m, n,
                    alpha,
                    x, incX,
                    y, incY,
                    A, ldA);
}

// dgeru
inline cublasStatus_t
geru(cublasHandle_t handle,
     int m, int n,
     const double* alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasDger");

  return cublasDger(handle,
                    m, n,
                    alpha,
                    x, incX,
                    y, incY,
                    A, ldA);
}

// cgeru
inline cublasStatus_t
geru(cublasHandle_t handle,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCgeru");

  return cublasCgeru(handle,
                     m, n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<const cuFloatComplex*>(y), incY,
                     reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zgeru
inline cublasStatus_t
geru(cublasHandle_t handle,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZgeru");

  return cublasZgeru(handle,
                     m, n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<const cuDoubleComplex*>(y), incY,
                     reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

// sgerc
inline cublasStatus_t
gerc(cublasHandle_t handle,
     int m, int n,
     const float* alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA)
{
  return geru(handle,
              m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// dgerc
inline cublasStatus_t
gerc(cublasHandle_t handle,
     int m, int n,
     const double* alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA)
{
  return geru(handle,
              m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// cgerc
inline cublasStatus_t
gerc(cublasHandle_t handle,
     int m, int n,
     const ComplexFloat* alpha,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasCgerc");

  return cublasCgerc(handle,
                     m, n,
                     reinterpret_cast<const cuFloatComplex*>(alpha),
                     reinterpret_cast<const cuFloatComplex*>(x), incX,
                     reinterpret_cast<const cuFloatComplex*>(y), incY,
                     reinterpret_cast<cuFloatComplex*>(A), ldA);
}

// zgerc
inline cublasStatus_t
gerc(cublasHandle_t handle,
     int m, int n,
     const ComplexDouble* alpha,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* A, int ldA)
{
  BLAM_DEBUG_OUT("cublasZgerc");

  return cublasZgerc(handle,
                     m, n,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(x), incX,
                     reinterpret_cast<const cuDoubleComplex*>(y), incY,
                     reinterpret_cast<cuDoubleComplex*>(A), ldA);
}

// sger
inline cublasStatus_t
ger(cublasHandle_t handle,
    int m, int n,
    const float* alpha,
    const float* x, int incX,
    const float* y, int incY,
    float* A, int ldA)
{
  return geru(handle,
              m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// dger
inline cublasStatus_t
ger(cublasHandle_t handle,
    int m, int n,
    const double* alpha,
    const double* x, int incX,
    const double* y, int incY,
    double* A, int ldA)
{
  return geru(handle,
              m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// cger
inline cublasStatus_t
ger(cublasHandle_t handle,
    int m, int n,
    const ComplexFloat* alpha,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    ComplexFloat* A, int ldA)
{
  return gerc(handle,
              m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// zger
inline cublasStatus_t
ger(cublasHandle_t handle,
    int m, int n,
    const ComplexDouble* alpha,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    ComplexDouble* A, int ldA)
{
  return gerc(handle,
              m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
inline auto
geru(const execution_policy<DerivedPolicy>& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(geru(handle(derived_cast(exec)),
                     m, n, &alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return geru(handle(derived_cast(exec)),
              m, n, &alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
inline auto
geru(const execution_policy<DerivedPolicy>& exec,
     Layout order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(geru(exec, m, n,
                     alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  if (order == ColMajor) {
    return geru(exec, m, n,
                alpha,
                x, incX,
                y, incY,
                A, ldA);
  } else { // RowMajor: swap m <=> n, x <=> y
    return geru(exec, n, m,
                alpha,
                y, incY,
                x, incX,
                A, ldA);
  }
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
inline auto
gerc(const execution_policy<DerivedPolicy>& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(gerc(handle(derived_cast(exec)),
                     m, n, &alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  return gerc(handle(derived_cast(exec)),
              m, n, &alpha,
              x, incX,
              y, incY,
              A, ldA);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
inline auto
gerc(const execution_policy<DerivedPolicy>& exec,
     Layout order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA)
    -> decltype(gerc(exec, m, n,
                     alpha,
                     x, incX,
                     y, incY,
                     A, ldA))
{
  if (order == RowMajor) {
    if ((std::is_same<MA,ComplexFloat>::value || std::is_same<MA,ComplexDouble>::value)) {
      // No zero-overhead solution exists for RowMajor+Complex gerc. Options are:
      // 0) Fail with return code, assert, or throw
      // 1) Decay to many dot/axpy
      // 2) Copy and conjugate y on input
      // 3) Promote to gemm

      // Here, we've chosen (3), which works when incX > 0 and incY > 0

      //assert(false && "No cublas::gerc for RowMajor");
      //return CUBLAS_STATUS_INVALID_VALUE;

      MA beta = 1;
      return gemm(exec, order, NoTrans, ConjTrans,
                  m, n, 1,
                  alpha,
                  x, incX,
                  y, incY,
                  beta,
                  A, ldA);
    }
  }

  return gerc(exec, m, n,
              alpha,
              x, incX,
              y, incY,
              A, ldA);
}

} // end namespace cublas
} // end namespace blam
