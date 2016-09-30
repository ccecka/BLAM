#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/static_const.h>

namespace blam
{
namespace adl
{
namespace detail
{

struct _dot
{
  template <typename... T>
  void operator()(T&&... t) const;
};

struct _dotu
{
  template <typename... T>
  void operator()(T&&... t) const;
};

struct _dotc
{
  template <typename... T>
  void operator()(T&&... t) const;
};

} // end namespace detail

// blam::adl::dot is a global function object
namespace
{
static const auto dot = detail::static_const<detail::_dot>::value;
}

// blam::adl::dotu is a global function object
namespace
{
static const auto dotu = detail::static_const<detail::_dotu>::value;
}

// blam::adl::dotc is a global function object
namespace
{
static const auto dotc = detail::static_const<detail::_dotc>::value;
}

} // end namespace adl
} // end namespace blam

#include <blam/adl/detail/dot.inl>
