#pragma once

#include <blam/detail/config.h>

#include <type_traits>

namespace blam
{
namespace detail
{

template <typename T>
struct is_execution_policy;

// execution_policy_base serves as a guard against
// inifinite recursion in blam entry points:
//
// template <typename DerivedPolicy>
// void foo(const blam::detail::execution_policy_base<DerivedPolicy> &s)
// {
//   using blam::system::detail::generic::foo;
//
//   foo(blam::detail::derived_cast(blam::detail::strip_const(s));
// }
//
// foo is not recursive when
// 1. DerivedPolicy is derived from blam::execution_policy below
// 2. generic::foo takes blam::execution_policy as a parameter
template <typename DerivedPolicy>
struct execution_policy_base {};


template <typename DerivedPolicy>
inline execution_policy_base<DerivedPolicy>&
strip_const(const execution_policy_base<DerivedPolicy>& x)
{
  return const_cast<execution_policy_base<DerivedPolicy>&>(x);
}


template <typename DerivedPolicy>
inline DerivedPolicy&
derived_cast(execution_policy_base<DerivedPolicy>& x)
{
  return static_cast<DerivedPolicy&>(x);
}


template <typename DerivedPolicy>
inline const DerivedPolicy&
derived_cast(const execution_policy_base<DerivedPolicy> &x)
{
  return static_cast<const DerivedPolicy&>(x);
}

} // end namespace detail


template <typename DerivedPolicy>
struct execution_policy
    : blam::detail::execution_policy_base<DerivedPolicy>
{};


} // end namespace blam
