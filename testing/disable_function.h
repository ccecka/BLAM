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

#define __REQUIRES(...) typename std::enable_if<__VA_ARGS__>::type* = nullptr

namespace experimental
{

template <class... T>
struct type_list {};


template <class T, class Set>
struct is_member;

template <class T>
struct is_member<T, type_list<>> : std::false_type {};

template <class T, class... Ss>
struct is_member<T, type_list<T,Ss...>> : std::true_type {};

template <class T, class S, class... Ss>
struct is_member<T, type_list<S,Ss...>> : is_member<T,type_list<Ss...>> {};


template <class ExecutionPolicy, class Disabled>
struct disabled_execution_policy : public ExecutionPolicy
{
  ExecutionPolicy& base() { return *this; }
  const ExecutionPolicy& base() const { return *this; }
};

template <class... CP, class ExecutionPolicy>
const disabled_execution_policy<ExecutionPolicy, type_list<CP...> >&
remove_customization_point(const ExecutionPolicy& policy)
{
  return *reinterpret_cast<const disabled_execution_policy<ExecutionPolicy, type_list<CP...>>*>(&policy);
}


template <class Function,
          class DerivedPolicy,
          class Disabled,
          class... Args,
          __REQUIRES(!is_member<Function,Disabled>::value)>
auto
invoke(const disabled_execution_policy<DerivedPolicy,Disabled>& exec, Function f, Args&&... args) ->
    decltype(f(exec.base(), std::forward<Args>(args)...))
{
  return f(exec.base(), std::forward<Args>(args)...);
}

template <class Function,
          class DerivedPolicy,
          class Disabled,
          class... Args,
          __REQUIRES(is_member<Function,Disabled>::value)>
void
invoke(const disabled_execution_policy<DerivedPolicy,Disabled>& exec, Function f, Args&&... args) = delete;

} // end namespace experimental

#undef __REQUIRES
