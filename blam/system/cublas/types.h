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

cublasOperation_t
cublas_type(Op trans) {
  switch (trans) {
    case Op::NoTrans:   return CUBLAS_OP_N;
    case Op::Trans:     return CUBLAS_OP_T;
    case Op::ConjTrans: return CUBLAS_OP_C;
    default:
      assert(false && "Invalid Op Parameter");
      return CUBLAS_OP_N;
  }
}

cublasFillMode_t
cublas_type(Uplo uplo) {
  switch (uplo) {
    case Uplo::Upper: return CUBLAS_FILL_MODE_UPPER;
    case Uplo::Lower: return CUBLAS_FILL_MODE_LOWER;
    default:
      assert(false && "Invalid Uplo Parameter");
      return CUBLAS_FILL_MODE_UPPER;
  }
}

cublasSideMode_t
cublas_type(Side side) {
  switch (side) {
    case Side::Left:  return CUBLAS_SIDE_LEFT;
    case Side::Right: return CUBLAS_SIDE_RIGHT;
    default:
      assert(false && "Invalid Side Parameter");
      return CUBLAS_SIDE_LEFT;
  }
}

cublasDiagType_t
cublas_type(Diag diag) {
  switch (diag) {
    case Diag::Unit:    return CUBLAS_DIAG_UNIT;
    case Diag::NonUnit: return CUBLAS_DIAG_NON_UNIT;
    default:
      assert(false && "Invalid Diag Parameter");
      return CUBLAS_DIAG_UNIT;
  }
}

void
check_status(cublasStatus_t status)
{
  if (status==CUBLAS_STATUS_SUCCESS) {
    return;
  }

  if (status==CUBLAS_STATUS_NOT_INITIALIZED) {
    std::cerr << "CUBLAS: Library was not initialized!" << std::endl;
  } else if  (status==CUBLAS_STATUS_INVALID_VALUE) {
    std::cerr << "CUBLAS: Parameter had illegal value!" << std::endl;
  } else if  (status==CUBLAS_STATUS_MAPPING_ERROR) {
    std::cerr << "CUBLAS: Error accessing GPU memory!" << std::endl;
  } else if  (status==CUBLAS_STATUS_ALLOC_FAILED) {
    std::cerr << "CUBLAS: allocation failed!" << std::endl;
  } else if  (status==CUBLAS_STATUS_ARCH_MISMATCH) {
    std::cerr << "CUBLAS: Device does not support double precision!" << std::endl;
  } else if  (status==CUBLAS_STATUS_EXECUTION_FAILED) {
    std::cerr << "CUBLAS: Failed to launch function on the GPU" << std::endl;
  } else if  (status==CUBLAS_STATUS_INTERNAL_ERROR) {
    std::cerr << "CUBLAS: An internal operation failed" << std::endl;
  } else {
    std::cerr << "CUBLAS: Unkown error" << std::endl;
  }

  assert(status==CUBLAS_STATUS_SUCCESS); // false
}

} // end namespace blam
