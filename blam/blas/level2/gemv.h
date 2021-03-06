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

#include <blam/detail/config.h>
#include <blam/adl/detail/customization_point.h>

BLAM_CUSTOMIZATION_POINT(gemv)

namespace blam
{

// Backend entry point
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
generic(blam::gemv_t, const ExecutionPolicy& exec,
        Layout order, Op trans,
        int m, int n,
        const Alpha& alpha,
        const MA* A, int ldA,
        const VX* x, int incX,
        const Beta& beta,
        VY* y, int incY) = delete;

// Default ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
generic(blam::gemv_t, const ExecutionPolicy& exec,
        Op trans,
        int m, int n,
        const Alpha& alpha,
        const MA* A, int ldA,
        const VX* x, int incX,
        const Beta& beta,
        VY* y, int incY)
BLAM_DECLTYPE_AUTO_RETURN
(
  blam::gemv(exec, ColMajor, trans,
             m, n,
             alpha,
             A, ldA,
             x, incX,
             beta,
             y, incY)
)

// Default NoTrans
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
auto
generic(blam::gemv_t, const ExecutionPolicy& exec,
        int m, int n,
        const Alpha& alpha,
        const MA* A, int ldA,
        const VX* x, int incX,
        const Beta& beta,
        VY* y, int incY)
BLAM_DECLTYPE_AUTO_RETURN
(
  blam::gemv(exec, NoTrans,
             m, n,
             alpha,
             A, ldA,
             x, incX,
             beta,
             y, incY)
)

} // end namespace blam
