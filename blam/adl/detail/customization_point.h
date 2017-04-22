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

#include <utility>  // std::forward

#include <blam/adl/detail/static_const.h>

#define BLAM_CUSTOMIZATION_POINT(NAME)                                         \
  namespace detail {                                                           \
  namespace generic = blam::system::generic;                                   \
                                                                               \
  struct _##NAME {                                                             \
   private:                                                                    \
    template <class... T,                                                      \
              class R = decltype(NAME (std::declval<T>()...))>                 \
    static constexpr R impl(int, T&&... t) {                                   \
      return NAME (std::forward<T>(t)...);                                     \
    }                                                                          \
                                                                               \
    template <class... T,                                                      \
              class R = decltype(generic::NAME (std::declval<T>()...))>        \
    static constexpr R impl(long, T&&... t) {                                  \
      return generic::NAME (std::forward<T>(t)...);                            \
    }                                                                          \
   public:                                                                     \
    template <typename... T>                                                   \
    constexpr auto operator()(T&&... t) const                                  \
        -> decltype(impl(42, std::forward<T>(t)...)) {                         \
      return impl(42, std::forward<T>(t)...);                                  \
    }                                                                          \
  };                                                                           \
                                                                               \
  }                                                                            \
                                                                               \
  namespace {                                                                  \
  constexpr auto const& NAME = detail::static_const<detail::_##NAME>::value;   \
  }
