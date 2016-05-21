#pragma once

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/batch_gemm.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag batch_gemm(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto _batch_gemm(T&&... t)
    -> has_an_adl<decltype(batch_gemm(std::forward<T>(t)...))>
{
  batch_gemm(std::forward<T>(t)...);
}

template <typename... T>
auto _batch_gemm(T&&... t)
    -> has_no_adl<decltype(batch_gemm(std::forward<T>(t)...))>
{
  blam::system::generic::batch_gemm(std::forward<T>(t)...);
}
#endif


template <typename R>
struct batch_gemm_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return batch_gemm(std::forward<T>(t)...);
  }
};

template <>
struct batch_gemm_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::batch_gemm(std::forward<T>(t)...))
  {
    return blam::system::generic::batch_gemm(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_batch_gemm::operator()(T&&... t) const
{
  batch_gemm_fn<decltype(batch_gemm(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
