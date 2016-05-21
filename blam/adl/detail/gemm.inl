#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/gemm.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag gemm(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto
_gemm(T&&... t) -> has_an_adl<decltype(gemm(std::forward<T>(t)...))>
{
  gemm(std::forward<T>(t)...);
}

template <typename... T>
auto
_gemm(T&&... t) -> has_no_adl<decltype(gemm(std::forward<T>(t)...))>
{
  blam::system::generic::gemm(std::forward<T>(t)...);
}
#endif


template <typename R>
struct gemm_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return gemm(std::forward<T>(t)...);
  }
};

template <>
struct gemm_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::gemm(std::forward<T>(t)...))
  {
    return blam::system::generic::gemm(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_gemm::operator()(T&&... t) const
{
  gemm_fn<decltype(gemm(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
