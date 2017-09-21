/******************************************************************************
 * Copyright (C) 2016-2017, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
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

#include <utility>

#include <blam/adl/detail/multi_function.h>
#include <blam/adl/detail/static_const.h>
#include <blam/invoke.h>

namespace blam
{
namespace detail
{

template <class Derived, class... Functions>
class customization_point : multi_function<Functions...>
{
 private:
  using super_t = multi_function<Functions...>;
  using derived_type = Derived;

  const derived_type& self() const
  {
    return static_cast<const derived_type&>(*this);
  }

 public:
  constexpr customization_point() = default;

  constexpr customization_point(Functions... funcs)
      : super_t(funcs...)
  {}

  /*
  template <class... Args>
  constexpr auto operator()(Args&&... args) const ->
      decltype(super_t::operator()(self(), std::forward<Args>(args)...))
  {
    return super_t::operator()(self(), std::forward<Args>(args)...);
  }
  */

  // NVCC EDG Bug Workaround
  template <class... Args>
  constexpr auto operator()(Args&&... args) const ->
      decltype(static_cast<const super_t&>(*this)(self(), std::forward<Args>(args)...))
  {
    return static_cast<const super_t&>(*this)(self(), std::forward<Args>(args)...);
  }
};

} // end namespace detail
} // end namespace blam

// A BLAM Customization Point is a forward-facing, extendable function
// provided by BLAM that:
// 1. Is a Neibler-style customization point and is always the BLAM entry-point.
// 2. When a Customization Point is called like a function, it:
//   a. First, tries to call the Customization Point by name as a member function
//       arg1.customization-point(args...)
//      I.e. if arg1 is an execution policy that provides this member function
//   b. Second, tries to call the Customization Point as a free function
//       customization-point(arg1, args...)
//       blam::invoke(arg1, customization-point, args...)
//      I.e. with ADL the arguments will determine the appropriate namespace
//      Note invoke(...) is considered equally to the free function.
//   c. Finally, tries to call a 'generic' fallback function
//       generic(customization-point, args...)
//      The generic fallbacks implement general argument defaults and transforms.

// This implementation works around an NVCC EDG Bug by defining the
// call_ functors in a separate namespace, which should not be necessary.

#define BLAM_CUSTOMIZATION_POINT(NAME)                                  \
  namespace blamadl {                                                   \
  struct call_member_##NAME {                                           \
    template <class CP, class Arg1, class... Args>                      \
    constexpr auto operator()(CP&&, Arg1&& arg1, Args&&... args) const -> \
        decltype(std::forward<Arg1>(arg1).NAME(std::forward<Args>(args)...)) \
    {                                                                   \
      return std::forward<Arg1>(arg1).NAME(std::forward<Args>(args)...); \
    }                                                                   \
  };                                                                    \
                                                                        \
  struct call_free_##NAME {                                             \
    template <class CP, class Arg1, class... Args>                      \
    constexpr auto operator()(CP&& cp, Arg1&& arg1, Args&&... args) const -> \
        decltype(blam::invoke(std::forward<Arg1>(arg1), std::forward<CP>(cp), std::forward<Args>(args)...)) \
    {                                                                   \
      return blam::invoke(std::forward<Arg1>(arg1), std::forward<CP>(cp), std::forward<Args>(args)...); \
    }                                                                   \
                                                                        \
    template <class CP, class... Args>                                  \
    constexpr auto operator()(CP&&, Args&&... args) const ->            \
        decltype(NAME(std::forward<Args>(args)...))                     \
    {                                                                   \
      return NAME(std::forward<Args>(args)...);                         \
    }                                                                   \
  };                                                                    \
                                                                        \
  struct call_generic_##NAME {                                          \
    template <class CP, class... Args>                                  \
    constexpr auto operator()(CP&& cp, Args&&... args) const ->         \
        decltype(generic(std::forward<CP>(cp), std::forward<Args>(args)...)) \
    {                                                                   \
      return generic(std::forward<CP>(cp), std::forward<Args>(args)...); \
    }                                                                   \
  };                                                                    \
  }                                                                     \
                                                                        \
  namespace blam {                                                      \
  struct NAME##_t : detail::customization_point<NAME##_t,               \
                                                blamadl::call_member_##NAME, \
                                                blamadl::call_free_##NAME, \
                                                blamadl::call_generic_##NAME> \
  {};                                                                   \
  namespace {                                                           \
  constexpr auto const& NAME = detail::static_const<NAME##_t>::value;   \
  }                                                                     \
  }
