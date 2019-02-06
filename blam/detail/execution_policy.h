/******************************************************************************
 * Copyright (C) 2016-2019, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#pragma once

#include <blam/detail/config.h>

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
