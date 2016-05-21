#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/static_const.h>

namespace blam
{
namespace adl
{
namespace detail
{

struct _copy
{
  template <typename... T>
  void operator()(T&&... t) const;
};

} // end namespace detail

// blam::adl::copy is a global function object
namespace
{
static const auto copy = detail::static_const<detail::_copy>::value;
}

} // end namespace adl
} // end namespace blam

#include <blam/adl/detail/copy.inl>
