#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/static_const.h>

namespace blam
{
namespace adl
{
namespace detail
{

struct _gemm
{
  template <typename... T>
  void operator()(T&&... t) const;
};

} // end namespace detail

// blam::adl::gemm is a global function object
namespace
{
static const auto gemm = detail::static_const<detail::_gemm>::value;
}

} // end namespace adl
} // end namespace blam

#include <blam/adl/detail/gemm.inl>
