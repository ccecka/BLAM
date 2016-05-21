#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/static_const.h>

namespace blam
{
namespace adl
{
namespace detail
{

struct _batch_gemm
{
  template <typename... T>
  void operator()(T&&... t) const;
};

} // end namespace detail

// blam::adl::batch_gemm is a global function object
namespace
{
static const auto batch_gemm = detail::static_const<detail::_batch_gemm>::value;
}

} // end namespace adl
} // end namespace blam

#include <blam/adl/detail/batch_gemm.inl>
