#pragma once

#include <blam/detail/config.h>
#include <blam/adl/dot.h>

namespace blam
{
namespace system
{
namespace generic
{

template <typename ExecutionPolicy,
          typename T, typename U>
void
dot(const ExecutionPolicy& exec,
    int n,
    const T* x, int incX,
    const T* y, int incY,
    U& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

template <typename ExecutionPolicy,
          typename T, typename U>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const T* x, int incX,
     const T* y, int incY,
     U& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename T, typename U>
void
dot(const ExecutionPolicy& exec,
    int n,
    const T* x,
    const T* y,
    U& result)
{
  blam::adl::dot(exec, n, x, 1, y, 1, result);
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename T, typename U>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const T* x,
     const T* y,
     U& result)
{
  blam::adl::dotu(exec, n, x, 1, y, 1, result);
}

// sdotu -> sdot
template <typename ExecutionPolicy>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const float* x, int incX,
     const float* y, int incY,
     float& result)
{
  blam::adl::dot(exec, n, x, incX, y, incY, result);
}

// ddotu -> ddot
template <typename ExecutionPolicy>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const double* x, int incX,
     const double* y, int incY,
     double& result)
{
  blam::adl::dot(exec, n, x, incX, y, incY, result);
}

} // end namespace generic
} // end namespace system
} // end namespace blam
