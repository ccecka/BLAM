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

#include <blam/detail/config.h>
#include <blam/adl/detail/detail/has_adl.h>

#include <blam/system/generic/dot.h>

#include <utility>  // std::forward

namespace blam
{
namespace adl
{
namespace detail
{

tag dot(...);
tag dotu(...);
tag dotc(...);

/** XXX WAR: nvcc/edg bug? **/
#if 0
template <typename... T>
auto _dot(T&&... t)
    -> has_an_adl<decltype(dot(std::forward<T>(t)...))>
{
  dot(std::forward<T>(t)...);
}

template <typename... T>
auto _dot(T&&... t)
    -> has_no_adl<decltype(dot(std::forward<T>(t)...))>
{
  blam::system::generic::dot(std::forward<T>(t)...);
}

template <typename... T>
auto _dotu(T&&... t)
    -> has_an_adl<decltype(dotu(std::forward<T>(t)...))>
{
  dotu(std::forward<T>(t)...);
}

template <typename... T>
auto _dotu(T&&... t)
    -> has_no_adl<decltype(dotu(std::forward<T>(t)...))>
{
  blam::system::generic::dotu(std::forward<T>(t)...);
}
#endif


template <typename R>
struct dot_fn
{
  template <typename... T>
  static R call(T&&... t)
  {
    return dot(std::forward<T>(t)...);
  }
};

template <>
struct dot_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::dot(std::forward<T>(t)...))
  {
    return blam::system::generic::dot(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_dot::operator()(T&&... t) const {
  dot_fn<decltype(dot(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct dotu_fn
{
  template <typename... T>
  static R call(T&&... t) {
    return dotu(std::forward<T>(t)...);
  }
};

template <>
struct dotu_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::dotu(std::forward<T>(t)...))
  {
    return blam::system::generic::dotu(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_dotu::operator()(T&&... t) const {
  dotu_fn<decltype(dotu(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

template <typename R>
struct dotc_fn
{
  template <typename... T>
  static R call(T&&... t) {
    return dotc(std::forward<T>(t)...);
  }
};

template <>
struct dotc_fn<tag>
{
  template <typename... T>
  static auto call(T&&... t)
      -> decltype(blam::system::generic::dotc(std::forward<T>(t)...))
  {
    return blam::system::generic::dotc(std::forward<T>(t)...);
  }
};

template <typename... T>
void
_dotc::operator()(T&&... t) const {
  dotc_fn<decltype(dotc(std::declval<T>()...))>::call(std::forward<T>(t)...);
}

} // end namespace detail
} // end namespace adl
} // end namespace blam
