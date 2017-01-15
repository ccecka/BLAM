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
#include <blam/detail/execution_policy.h>

#include <cuda.h>
#include <cublas_v2.h>

namespace blam
{
namespace cublas
{

// this awkward sequence of definitions arise
// from the desire both for tag to derive
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template <typename>
struct execution_policy;

// specialize execution_policy for tag
template <>
struct execution_policy<tag>
    : blam::execution_policy<tag>
{};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag> {};

// allow conversion to tag when it is not a successor
template <typename Derived>
struct execution_policy
    : blam::execution_policy<Derived>
{
  // allow conversion to tag
  inline operator tag () const {
    return tag();
  }
};

// given any old execution_policy, we return the default handle
template <typename DerivedPolicy>
inline cublasHandle_t
handle(const execution_policy<DerivedPolicy>& exec)
{
  static_assert(sizeof(DerivedPolicy) == 0, "BLAM UNIMPLEMENTED");
  return {};
}

// base class for execute_on_handle
template <typename DerivedPolicy>
struct execute_on_handle_base
    : public blam::cublas::execution_policy<DerivedPolicy>
{
  execute_on_handle_base(void) {}

  execute_on_handle_base(const cublasHandle_t& handle)
      : m_handle(handle)
  {}

  DerivedPolicy with(const cublasHandle_t& h) const
  {
    // create a copy of *this to return
    // make sure it is the derived type
    DerivedPolicy result = blam::detail::derived_cast(*this);

    // change the result's handle to h
    result.set_handle(h);

    return result;
  }

  DerivedPolicy on(const cudaStream_t& s) const
  {
    // create a copy of *this to return
    // make sure it is the derived type
    DerivedPolicy result = blam::detail::derived_cast(*this);

    // change the result's handle to s
    result.set_stream(s);

    return result;
  }

 private:
  // handle() is a friend function because we call it through ADL
  friend inline cublasHandle_t handle(const execute_on_handle_base& exec)
  {
    return exec.m_handle;
  }

  inline void set_handle(const cublasHandle_t& h)
  {
    m_handle = h;
  }

  inline void set_stream(const cudaStream_t& s)
  {
    cublasSetStream(m_handle, s);
  }

  cublasHandle_t m_handle;
};


// execution policy which submits kernel launches on a given handle
class execute_on_handle
    : public execute_on_handle_base<execute_on_handle>
{
  typedef execute_on_handle_base<execute_on_handle> super_t;

 public:
  // XXX: Default handle??
  inline execute_on_handle(void) {}

  inline execute_on_handle(const cublasHandle_t& handle)
      : super_t(handle)
  {}
};

static const execute_on_handle par;

} // end namespace cublas


cublasOperation_t
cublas_type(Transpose trans) {
  switch (trans) {
    case NoTrans:   return CUBLAS_OP_N;
    case Trans:     return CUBLAS_OP_T;
    case ConjTrans: return CUBLAS_OP_C;
    default:        assert(false && "Invalid Transpose Parameter"); return CUBLAS_OP_N;
  }
}

cublasFillMode_t
cublas_type(StorageUpLo uplo) {
  switch (uplo) {
    case Upper: return CUBLAS_FILL_MODE_UPPER;
    case Lower: return CUBLAS_FILL_MODE_LOWER;
    default: assert(false && "Invalid StorageUpLo Parameter"); return CUBLAS_FILL_MODE_UPPER;
  }
}

cublasSideMode_t
cublas_type(Side side) {
  switch (side) {
    case Left:  return CUBLAS_SIDE_LEFT;
    case Right: return CUBLAS_SIDE_RIGHT;
    default: assert(false && "Invalid Side Parameter"); return CUBLAS_SIDE_LEFT;
  }
}

cublasDiagType_t
cublas_type(Diag diag) {
  switch (diag) {
    case Unit:    return CUBLAS_DIAG_UNIT;
    case NonUnit: return CUBLAS_DIAG_NON_UNIT;
    default: assert(false && "Invalid Diag Parameter"); return CUBLAS_DIAG_UNIT;
  }
}

void
checkStatus(cublasStatus_t status)
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
