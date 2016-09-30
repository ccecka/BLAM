#pragma once

#include <blam/detail/config.h>
#include <blam/adl/dot.h>

namespace blam
{

#if 0
template <typename ExecutionPolicy,
          typename T, typename U>
void
dot(ExecutionPolicy&& exec,
    int n,
    const T* x, int incX,
    const T* y, int incY,
    U& result);

template <typename ExecutionPolicy,
          typename T, typename U>
void
dot(ExecutionPolicy&& exec,
    int n,
    const T* x,
    const T* y,
    U& result);

template <typename ExecutionPolicy,
          typename T, typename U>
void
dotu(ExecutionPolicy&& exec,
     int n,
     const T* x, int incX,
     const T* y, int incY,
     U& result);

template <typename ExecutionPolicy,
          typename T, typename U>
void
dotu(ExecutionPolicy&& exec,
     int n,
     const T* x,
     const T* y,
     U& result);
#endif

using blam::adl::dot;
using blam::adl::dotu;
using blam::adl::dotc;

} // end namespace blam
