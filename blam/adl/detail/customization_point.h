#pragma once

#include <utility>  // std::forward

#include <blam/adl/detail/static_const.h>
#include <blam/adl/detail/has_adl.h>

#define BLAM_CUSTOMIZATION_POINT(NAME)                                  \
  namespace detail {                                                    \
  tag NAME (...);                                                       \
                                                                        \
  template <typename R>                                                 \
  struct NAME##_fn {                                                    \
    template <typename... T>                                            \
    static R call(T&&... t) {                                           \
      return NAME (std::forward<T>(t)...);                              \
    }                                                                   \
  };                                                                    \
                                                                        \
  template <>                                                           \
  struct NAME##_fn<tag> {                                               \
    template <typename... T>                                            \
    static auto call(T&&... t)                                          \
        -> decltype(blam::system::generic::NAME(std::forward<T>(t)...)) { \
      return blam::system::generic::NAME(std::forward<T>(t)...);        \
    }                                                                   \
  };                                                                    \
                                                                        \
  struct _##NAME {                                                      \
    template <typename... T>                                            \
    void operator()(T&&... t) const {                                   \
      NAME##_fn<decltype( NAME (std::declval<T>()...))>::call(std::forward<T>(t)...); \
    }                                                                   \
  };                                                                    \
  }                                                                     \
                                                                        \
  namespace {                                                           \
  constexpr auto const& NAME = detail::static_const<detail::_##NAME>::value; \
  }


#if 0
/**
 * XXX TODO: This version is preferred, but exposes an EDG bug...
 **/
#define BLAM_CUSTOMIZATION_POINT(NAME)                                   \
  namespace detail {                                                     \
  tag NAME (...);                                                        \
                                                                         \
  struct _##NAME {                                                       \
    template <class... T>                                                \
    constexpr auto operator()(T&&... t) const                            \
        -> has_an_adl<decltype( NAME (std::forward<T>(t)...))> {         \
      return NAME (std::forward<T>(t)...);                               \
    }                                                                    \
                                                                         \
    template <class... T>                                                \
    constexpr auto operator()(T&&... t) const                            \
        -> has_no_adl<decltype( NAME (std::forward<T>(t)...)),           \
                      decltype(blam::system::generic::NAME (std::forward<T>(t)...))> { \
      return blam::system::generic::NAME (std::forward<T>(t)...);        \
    }                                                                    \
  };                                                                     \
  }                                                                      \
                                                                         \
  namespace {                                                            \
  constexpr auto const& NAME = detail::static_const<detail::_##NAME>::value; \
  }
#endif
