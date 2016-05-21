#pragma once

#include <blam/detail/config.h>
#include <blam/adl/copy.h>

namespace blam
{

#if 0
template <typename ExecutionPolicy,
          typename T, typename U>
void
copy(ExecutionPolicy&& exec,
     int n,
     const T* x, int incX,
     U* y, int incY);

template <typename ExecutionPolicy,
          typename T, typename U>
void
copy(ExecutionPolicy&& exec,
     int n,
     const T* x,
     U* y);
#endif

using blam::adl::copy;

} // end namespace blam
