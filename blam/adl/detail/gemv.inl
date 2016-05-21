#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/gemv.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag gemv(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto
_gemv(T&&... t) -> has_an_adl<decltype(gemv(std::forward<T>(t)...))>
{
  gemv(std::forward<T>(t)...);
}

template <typename... T>
auto
_gemv(T&&... t) -> has_no_adl<decltype(gemv(std::forward<T>(t)...))>
{
  blam::system::generic::gemv(std::forward<T>(t)...);
}
#endif


template <typename R>
struct gemv_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return gemv(std::forward<T>(t)...);
  }
};

template <>
struct gemv_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::gemv(std::forward<T>(t)...))
  {
    return blam::system::generic::gemv(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_gemv::operator()(T&&... t) const
{
  gemv_fn<decltype(gemv(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
