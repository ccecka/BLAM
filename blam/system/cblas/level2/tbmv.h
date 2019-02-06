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

#include <blam/system/cblas/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// stbmv
inline void
tbmv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
     int n, int k,
     const float* A, int ldA,
     float* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_stbmv");

  cblas_stbmv(order, uplo, trans, diag,
              n, k,
              A, ldA,
              x, incX);
}

// dtbmv
inline void
tbmv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
     int n, int k,
     const double* A, int ldA,
     double* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_dtbmv");

  cblas_dtbmv(order, uplo, trans, diag,
              n, k,
              A, ldA,
              x, incX);
}

// ctbmv
inline void
tbmv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
     int n, int k,
     const ComplexFloat* A, int ldA,
     ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_ctbmv");

  cblas_ctbmv(order, uplo, trans, diag,
              n, k,
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<float*>(x), incX);
}

// ztbmv
inline void
tbmv(CBLAS_LAYOUT order, CBLAS_UPLO uplo,
     CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
     int n, int k,
     const ComplexDouble* A, int ldA,
     ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_ztbmv");

  cblas_ztbmv(order, uplo, trans, diag,
              n, k,
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<double*>(x), incX);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename MA, typename VX>
inline auto
tbmv(const execution_policy<DerivedPolicy>& /*exec*/,
     Layout order, Uplo uplo, Op trans, Diag diag,
     int n, int k,
     const MA* A, int ldA,
     VX* x, int incX)
    -> decltype(tbmv(cblas_type(order), cblas_type(uplo),
                     cblas_type(trans), cblas_type(diag),
                     n, k,
                     A, ldA,
                     x, incX))
{
  return tbmv(cblas_type(order), cblas_type(uplo),
              cblas_type(trans), cblas_type(diag),
              n, k,
              A, ldA,
              x, incX);
}

} // end namespace cblas
} // end namespace blam
