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

namespace blam
{
namespace detail
{

template <class... Implementations>
class multi_function;

template <>
class multi_function<> {};

// a multi_function has several different implementations
// when called, the multi_function selects the first implementation that is not ill-formed
template <class Implementation1, class... Implementations>
class multi_function<Implementation1,Implementations...> : multi_function<Implementations...>
{
 private:
  using super_t = multi_function<Implementations...>;

  Implementation1 impl_;

  template <class... Args>
  static constexpr auto impl(const multi_function& self, Args&&... args) ->
      decltype(self.impl_(std::forward<Args>(args)...))
  {
    return self.impl_(std::forward<Args>(args)...);
  }

  template <class... Args>
  static constexpr auto impl(const super_t& super, Args&&... args) ->
      decltype(super(std::forward<Args>(args)...))
  {
    return super(std::forward<Args>(args)...);
  }

 public:
  constexpr multi_function() = default;

  constexpr multi_function(Implementation1 impl1, Implementations... impls)
      : multi_function<Implementations...>(impls...), impl_(impl1)
  {}

  template <class... Args>
  constexpr auto operator()(Args&&... args) const ->
      decltype(multi_function::impl(*this, std::forward<Args>(args)...))
  {
    return multi_function::impl(*this, std::forward<Args>(args)...);
  }
};

} // end namespace detail
} // end namespace blam
