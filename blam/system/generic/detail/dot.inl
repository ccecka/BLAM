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
          typename VX, typename VY, typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
{
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const VX* x,
    const VY* y,
    R& result)
{
  blam::adl::dotc(exec, n, x, 1, y, 1, result);
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const VX* x,
     const VY* y,
     R& result)
{
  blam::adl::dotc(exec, n, x, 1, y, 1, result);
}

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const VX* x,
     const VY* y,
     R& result)
{
  blam::adl::dotu(exec, n, x, 1, y, 1, result);
}

// sdot -> sdotu
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const float* x, int incX,
    const float* y, int incY,
    R& result)
{
  blam::adl::dotu(exec, n, x, incX, y, incY, result);
}

// sdotc -> sdotu
template <typename ExecutionPolicy,
          typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const float* x, int incX,
     const float* y, int incY,
     R& result)
{
  blam::adl::dotu(exec, n, x, incX, y, incY, result);
}

// ddot -> ddotu
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const double* x, int incX,
    const double* y, int incY,
    R& result)
{
  blam::adl::dotu(exec, n, x, incX, y, incY, result);
}

// ddotc -> ddotu
template <typename ExecutionPolicy,
          typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const double* x, int incX,
     const double* y, int incY,
     R& result)
{
  blam::adl::dotu(exec, n, x, incX, y, incY, result);
}

// cdot -> cdotc
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    R& result)
{
  blam::adl::dotc(exec, n, x, incX, y, incY, result);
}

// zdot -> zdotc
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    R& result)
{
  blam::adl::dotc(exec, n, x, incX, y, incY, result);
}

} // end namespace generic
} // end namespace system
} // end namespace blam
