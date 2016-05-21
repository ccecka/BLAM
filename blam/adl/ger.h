#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/static_const.h>

namespace blam
{
namespace adl
{
namespace detail
{

struct _ger
{
  template <typename... T>
  void operator()(T&&... t) const;
};

struct _geru
{
  template <typename... T>
  void operator()(T&&... t) const;
};

} // end namespace detail

// blam::adl::ger is a global function object
namespace
{
static const auto ger = detail::static_const<detail::_ger>::value;
}

// blam::adl::geru is a global function object
namespace
{
static const auto geru = detail::static_const<detail::_geru>::value;
}

} // end namespace adl
} // end namespace blam

#include <blam/adl/detail/ger.inl>
