#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/static_const.h>

namespace blam
{
namespace adl
{
namespace detail
{

struct _gemv
{
  template <typename... T>
  void operator()(T&&... t) const;
};

} // end namespace detail

// blam::adl::gemv is a global function object
namespace
{
static const auto gemv = detail::static_const<detail::_gemv>::value;
}

} // end namespace adl
} // end namespace blam

#include <blam/adl/detail/gemv.inl>
