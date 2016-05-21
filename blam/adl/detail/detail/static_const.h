#pragma once

namespace blam
{
namespace adl
{
namespace detail
{

template <typename T>
struct static_const
{
  static constexpr T value {};
};
template <typename T>
constexpr T static_const<T>::value;

} // end namespace detail
} // end namespace adl
} // end namespace blam
