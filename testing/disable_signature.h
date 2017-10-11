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

#include <type_traits>
#include <utility>  // std::forward, std::declval
#include <functional>

#define __REQUIRES(...) typename std::enable_if<__VA_ARGS__>::type* = nullptr

namespace experimental
{

template <class F, class... Args>
struct signature;

template <class... Conds>
struct or_ : std::false_type {};

template <class Cond, class... Conds>
struct or_<Cond, Conds...> : std::conditional<Cond::value, std::true_type, or_<Conds...>>::type {};

template <class T, class... Us>
struct is_any : or_<std::is_same<T,Us>...> {};

// Inherit from ExecutionPolicy's ParentPolicy to make the ParentPolicy's customization points available
template <class ExecutionPolicy, class ParentPolicy, class... Disabled>
class disabled_execution_policy : public ParentPolicy
{
 public:
  ParentPolicy& base() { return *this; }
  const ParentPolicy& base() const { return *this; }

 private:
  // this member ensures sizeof(disabled_execution_policy) >= sizeof(ExecutionPolicy)
  char padding_[sizeof(ExecutionPolicy) - sizeof(ParentPolicy)];
};

// There is no declared ParentPolicy, so inherit from ExecutionPolicy which will be disabled
template <class ExecutionPolicy, class... Disabled>
class disabled_execution_policy<ExecutionPolicy, void, Disabled...> : public ExecutionPolicy
{
 public:
  ExecutionPolicy& base() { return *this; }
  const ExecutionPolicy& base() const { return *this; }
};

template <class... CP, class ExecutionPolicy>
const disabled_execution_policy<ExecutionPolicy, void, CP...>&
remove_customization_point(const ExecutionPolicy& policy)
{
  return *reinterpret_cast<const disabled_execution_policy<ExecutionPolicy, void, CP...>*>(&policy);
}

template <class ParentPolicy, class... CP, class ExecutionPolicy>
const disabled_execution_policy<ExecutionPolicy, ParentPolicy, CP...>&
prefer_customization_point(const ExecutionPolicy& policy)
{
  return *reinterpret_cast<const disabled_execution_policy<ExecutionPolicy, ParentPolicy, CP...>*>(&policy);
}


template <class Function, class... Disabled,
          class ExecutionPolicy, class ParentPolicy,
          class... Args,
          __REQUIRES(!is_any<signature<Function,Args...>, Disabled...>::value)>
auto
invoke(const disabled_execution_policy<ExecutionPolicy, ParentPolicy, Disabled...>& policy,
       Function f, Args&&... args) ->
    decltype(f(policy.base(), std::forward<Args>(args)...))
{
  return f(policy.base(), std::forward<Args>(args)...);
}

template <class Function, class... Disabled,
          class ExecutionPolicy,
          class... Args,
          __REQUIRES(is_any<signature<Function,Args...>, Disabled...>::value)>
auto
invoke(const disabled_execution_policy<ExecutionPolicy, void, Disabled...>& policy,
       Function f, Args&&... args) ->
    decltype(f(policy.base(), std::forward<Args>(args)...)) = delete;

} // end namespace experimental

#undef __REQUIRES
