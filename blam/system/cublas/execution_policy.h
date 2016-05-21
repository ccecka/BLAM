#pragma once

#include <cublas_v2.h>

#include <blam/detail/config.h>
#include <blam/detail/execution_policy.h>

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
cublas_transpose(Transpose trans) {
  switch (trans) {
    case NoTrans:   return CUBLAS_OP_N;
    case Trans:     return CUBLAS_OP_T;
    case ConjTrans: return CUBLAS_OP_C;
    default:        assert(false); return CUBLAS_OP_N;
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

// Include algorithms
#include <blam/system/cublas/copy.h>
#include <blam/system/cublas/dot.h>

#include <blam/system/cublas/gemv.h>
#include <blam/system/cublas/ger.h>

#include <blam/system/cublas/gemm.h>

#include <blam/system/cublas/batch_gemm.h>
