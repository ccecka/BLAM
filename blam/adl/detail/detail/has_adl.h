#pragma once

#include <type_traits>

namespace blam
{
namespace adl
{
namespace detail
{

struct tag;

/** XXX WAR: nvcc bug?
template <typename T>
using has_an_adl = typename std::enable_if<! std::is_same<tag,T>::value>::type;

template <typename T>
using has_no_adl = typename std::enable_if<  std::is_same<tag,T>::value>::type;
*/

template <typename T>
using has_an_adl = std::enable_if<!std::is_same<tag,T>::value>;

template <typename T>
using has_no_adl = std::enable_if< std::is_same<tag,T>::value>;

}
}
}
