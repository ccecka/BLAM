#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/copy.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag copy(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto _copy(T&&... t)
    -> has_an_adl<decltype(copy(std::forward<T>(t)...))>
{
  copy(std::forward<T>(t)...);
}

template <typename... T>
auto _copy(T&&... t)
    -> has_no_adl<decltype(copy(std::forward<T>(t)...))>
{
  blam::system::generic::copy(std::forward<T>(t)...);
}
#endif


template <typename R>
struct copy_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return copy(std::forward<T>(t)...);
  }
};

template <>
struct copy_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::copy(std::forward<T>(t)...))
  {
    return blam::system::generic::copy(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_copy::operator()(T&&... t) const
{
  copy_fn<decltype(copy(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
