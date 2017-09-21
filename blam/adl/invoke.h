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

#include <blam/adl/detail/static_const.h>
#include <blam/adl/detail/multi_function.h>

namespace blamadl
{

struct member_function_invoke
{
  template <class Policy, class... Args>
  constexpr auto operator()(Policy&& policy, Args&&... args) const ->
      decltype(std::forward<Policy>(policy).invoke(std::forward<Args>(args)...))
  {
    return std::forward<Policy>(policy).invoke(std::forward<Args>(args)...);
  }
};

struct free_function_invoke
{
  template <class Policy, class... Args>
  constexpr auto operator()(Policy&& policy, Args&&... args) const ->
      decltype(invoke(std::forward<Policy>(policy), std::forward<Args>(args)...))
  {
    return invoke(std::forward<Policy>(policy), std::forward<Args>(args)...);
  }
};

struct invoke_function
{
  template <class Function, class... Args>
  constexpr auto operator()(Function&& f, Args&&... args) const ->
      decltype(std::forward<Function>(f)(std::forward<Args>(args)...))
  {
    return std::forward<Function>(f)(std::forward<Args>(args)...);
  }
};

} // end namespace blamadl

namespace blam
{
using invoke_t = detail::multi_function<blamadl::member_function_invoke,
                                        blamadl::free_function_invoke,
                                        blamadl::invoke_function>;
namespace {
constexpr auto const& invoke = detail::static_const<invoke_t>::value;
} // end anon namespace

} // end namespace blam
