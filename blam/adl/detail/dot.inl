#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/dot.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag dot(...);
tag dotu(...);
tag dotc(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto _dot(T&&... t)
    -> has_an_adl<decltype(dot(std::forward<T>(t)...))>
{
  dot(std::forward<T>(t)...);
}

template <typename... T>
auto _dot(T&&... t)
    -> has_no_adl<decltype(dot(std::forward<T>(t)...))>
{
  blam::system::generic::dot(std::forward<T>(t)...);
}

template <typename... T>
auto _dotu(T&&... t)
    -> has_an_adl<decltype(dotu(std::forward<T>(t)...))>
{
  dotu(std::forward<T>(t)...);
}

template <typename... T>
auto _dotu(T&&... t)
    -> has_no_adl<decltype(dotu(std::forward<T>(t)...))>
{
  blam::system::generic::dotu(std::forward<T>(t)...);
}
#endif


template <typename R>
struct dot_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return dot(std::forward<T>(t)...);
  }
};

template <>
struct dot_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::dot(std::forward<T>(t)...))
  {
    return blam::system::generic::dot(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_dot::operator()(T&&... t) const {
  dot_fn<decltype(dot(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct dotu_fn
{
  template <typename... T>
  static R call(T&&... t) {
    return dotu(std::forward<T>(t)...);
  }
};

template <>
struct dotu_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::dotu(std::forward<T>(t)...))
  {
    return blam::system::generic::dotu(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_dotu::operator()(T&&... t) const {
  dotu_fn<decltype(dotu(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct dotc_fn
{
  template <typename... T>
  static R call(T&&... t) {
    return dotc(std::forward<T>(t)...);
  }
};

template <>
struct dotc_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::dotc(std::forward<T>(t)...))
  {
    return blam::system::generic::dotc(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_dotc::operator()(T&&... t) const {
  dotc_fn<decltype(dotc(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
