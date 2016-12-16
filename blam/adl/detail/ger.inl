/******************************************************************************
 * Copyright (C) 2016, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016, NVIDIA CORPORATION.  All rights reserved.
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
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/ger.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag ger(...);
tag geru(...);
tag gerc(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto
_ger(T&&... t) -> has_an_adl<decltype(ger(std::forward<T>(t)...))>
{
  ger(std::forward<T>(t)...);
}

template <typename... T>
auto
_ger(T&&... t) -> has_no_adl<decltype(ger(std::forward<T>(t)...))>
{
  blam::system::generic::ger(std::forward<T>(t)...);
}
#endif


template <typename R>
struct ger_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return ger(std::forward<T>(t)...);
  }
};

template <>
struct ger_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::ger(std::forward<T>(t)...))
  {
    return blam::system::generic::ger(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_ger::operator()(T&&... t) const
{
  ger_fn<decltype(ger(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct geru_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return geru(std::forward<T>(t)...);
  }
};

template <>
struct geru_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::geru(std::forward<T>(t)...))
  {
    return blam::system::generic::geru(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_geru::operator()(T&&... t) const
{
  geru_fn<decltype(geru(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct gerc_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return gerc(std::forward<T>(t)...);
  }
};

template <>
struct gerc_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::gerc(std::forward<T>(t)...))
  {
    return blam::system::generic::gerc(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_gerc::operator()(T&&... t) const
{
  gerc_fn<decltype(gerc(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
