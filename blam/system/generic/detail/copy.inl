#pragma once

#include <blam/detail/config.h>
#include <blam/adl/copy.h>

namespace blam
{
namespace system
{
namespace generic
{

template <typename ExecutionPolicy,
          typename T, typename U>
void
copy(const ExecutionPolicy& exec,
     int n,
     const T* x, int incX,
     U* y, int incY)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename T, typename U>
void
copy(const ExecutionPolicy& exec,
     int n,
     const T* x,
     U* y)
{
  blam::adl::copy(exec, n, x, 1, y, 1);
}

} // end namespace generic
} // end namespace system
} // end namespace blam
