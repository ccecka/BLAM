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

namespace blam
{

CBLAS_LAYOUT
cblas_type(Layout order) {
  switch (order) {
    case Layout::ColMajor: return CblasColMajor;
    case Layout::RowMajor: return CblasRowMajor;
    default:
      assert(false && "Invalid Layout Parameter");
      return CblasColMajor;
  }
}

CBLAS_TRANSPOSE
cblas_type(Op trans) {
  switch (trans) {
    case Op::NoTrans:   return CblasNoTrans;
    case Op::Trans:     return CblasTrans;
    case Op::ConjTrans: return CblasConjTrans;
    default:
      assert(false && "Invalid Op Parameter");
      return CblasNoTrans;
  }
}

CBLAS_UPLO
cblas_type(Uplo uplo) {
  switch (uplo) {
    case Uplo::Upper: return CblasUpper;
    case Uplo::Lower: return CblasLower;
    default:
      assert(false && "Invalid Uplo Parameter");
      return CblasUpper;
  }
}

CBLAS_SIDE
cblas_type(Side side) {
  switch (side) {
    case Side::Left:  return CblasLeft;
    case Side::Right: return CblasRight;
    default:
      assert(false && "Invalid Side Parameter");
      return CblasLeft;
  }
}

CBLAS_DIAG
cblas_type(Diag diag) {
  switch (diag) {
    case Diag::Unit:    return CblasUnit;
    case Diag::NonUnit: return CblasNonUnit;
    default:
      assert(false && "Invalid Diag Parameter");
      return CblasUnit;
  }
}

} // end namespace blam
