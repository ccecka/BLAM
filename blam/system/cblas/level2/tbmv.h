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
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// stbmv
void
tbmv(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
     const CBLAS_TRANSPOSE transA, const CBLAS_DIAG diag,
     int n, int k,
     const float* A, int ldA,
     float* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_stbmv");

  cblas_stbmv(order, upLo, transA, diag,
              n, k,
              A, ldA,
              x, incX);
}

// dtbmv
void
tbmv(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
     const CBLAS_TRANSPOSE transA, const CBLAS_DIAG diag,
     int n, int k,
     const double* A, int ldA,
     double* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_dtbmv");

  cblas_dtbmv(order, upLo, transA, diag,
              n, k,
              A, ldA,
              x, incX);
}

// ctbmv
void
tbmv(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
     const CBLAS_TRANSPOSE transA, const CBLAS_DIAG diag,
     int n, int k,
     const ComplexFloat* A, int ldA,
     ComplexFloat* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_ctbmv");

  cblas_ctbmv(order, upLo, transA, diag,
              n, k,
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<float*>(x), incX);
}

// ztbmv
void
tbmv(const CBLAS_LAYOUT order, const CBLAS_UPLO upLo,
     const CBLAS_TRANSPOSE transA, const CBLAS_DIAG diag,
     int n, int k,
     const ComplexDouble* A, int ldA,
     ComplexDouble* x, int incX)
{
  BLAM_DEBUG_OUT("cblas_ztbmv");

  cblas_ztbmv(order, upLo, transA, diag,
              n, k,
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<double*>(x), incX);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename MA, typename VX>
auto
tbmv(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, StorageUpLo upLo, Transpose transA, Diag diag,
     int n, int k,
     const MA* A, int ldA,
     VX* x, int incX)
    -> decltype(tbmv(cblas_type(order), cblas_type(upLo),
                     cblas_type(transA), cblas_type(diag),
                     n, k,
                     A, ldA,
                     x, incX))
{
  return tbmv(cblas_type(order), cblas_type(upLo),
              cblas_type(transA), cblas_type(diag),
              n, k,
              A, ldA,
              x, incX);
}

} // end namespace cblas
} // end namespace blam
