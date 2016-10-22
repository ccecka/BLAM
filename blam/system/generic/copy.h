#pragma once

#include <blam/detail/config.h>
#include <blam/system/generic/execution_policy.h>

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
     U* y, int incY);

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename T, typename U>
void
copy(const ExecutionPolicy& exec,
     int n,
     const T* x,
     U* y);

} // end namespace generic
} // end namespace system
} // end namespace blam

#include <blam/system/generic/detail/copy.inl>
