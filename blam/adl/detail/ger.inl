#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/ger.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag ger(...);
tag geru(...);
tag gerc(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto
_ger(T&&... t) -> has_an_adl<decltype(ger(std::forward<T>(t)...))>
{
  ger(std::forward<T>(t)...);
}

template <typename... T>
auto
_ger(T&&... t) -> has_no_adl<decltype(ger(std::forward<T>(t)...))>
{
  blam::system::generic::ger(std::forward<T>(t)...);
}
#endif


template <typename R>
struct ger_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return ger(std::forward<T>(t)...);
  }
};

template <>
struct ger_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::ger(std::forward<T>(t)...))
  {
    return blam::system::generic::ger(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_ger::operator()(T&&... t) const
{
  ger_fn<decltype(ger(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct geru_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return geru(std::forward<T>(t)...);
  }
};

template <>
struct geru_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::geru(std::forward<T>(t)...))
  {
    return blam::system::generic::geru(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_geru::operator()(T&&... t) const
{
  geru_fn<decltype(geru(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct gerc_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return gerc(std::forward<T>(t)...);
  }
};

template <>
struct gerc_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::gerc(std::forward<T>(t)...))
  {
    return blam::system::generic::gerc(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_gerc::operator()(T&&... t) const
{
  gerc_fn<decltype(gerc(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
