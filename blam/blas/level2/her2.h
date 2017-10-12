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
#include <blam/adl/detail/customization_point.h>

BLAM_CUSTOMIZATION_POINT(her2);

#include <blam/blas/level2/syr2.h>

namespace blam
{

// Backend entry point
template <typename ExecutionPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
void
generic(blam::her2_t, const ExecutionPolicy& exec,
        Layout order, Uplo uplo,
        int n,
        const Alpha& alpha,
        const VX* x, int incX,
        const VY* y, int incY,
        MA* A, int ldA) = delete;

// Default ColMajor
template <typename ExecutionPolicy,
          typename Alpha,
          typename VX, typename VY, typename MA>
auto
generic(blam::her2_t, const ExecutionPolicy& exec,
        Uplo uplo,
        int n,
        const Alpha& alpha,
        const VX* x, int incX,
        const VY* y, int incY,
        MA* A, int ldA)
BLAM_DECLTYPE_AUTO_RETURN
(
  blam::her2(exec,
             ColMajor, uplo,
             n,
             alpha,
             x, incX,
             y, incY,
             A, ldA)
)

// sher2 -> syr2
template <typename ExecutionPolicy>
auto
generic(blam::her2_t, const ExecutionPolicy& exec,
        Layout order, Uplo uplo,
        int n,
        const float& alpha,
        const float* x, int incX,
        const float* y, int incY,
        float* A, int ldA)
BLAM_DECLTYPE_AUTO_RETURN
(
  blam::syr2(exec,
             order, uplo,
             n,
             alpha,
             x, incX,
             y, incY,
             A, ldA)
)

// dher2 -> syr2
template <typename ExecutionPolicy>
auto
generic(blam::her2_t, const ExecutionPolicy& exec,
        Layout order, Uplo uplo,
        int n,
        const double& alpha,
        const double* x, int incX,
        const double* y, int incY,
        double* A, int ldA)
BLAM_DECLTYPE_AUTO_RETURN
(
  blam::syr2(exec,
             order, uplo,
             n,
             alpha,
             x, incX,
             y, incY,
             A, ldA)
)

} // end namespace blam
