#pragma once

#include <blam/detail/config.h>

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
     R& result);

template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result);

template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const VX* x, int incX,
    const VY* y, int incY,
    R& result);

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const VX* x,
    const VY* y,
    R& result);

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const VX* x,
     const VY* y,
     R& result);

// incX,incY -> 1,1
template <typename ExecutionPolicy,
          typename VX, typename VY, typename R>
void
dotu(const ExecutionPolicy& exec,
     int n,
     const VX* x,
     const VY* y,
     R& result);

// sdot -> sdotu
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const float* x, int incX,
    const float* y, int incY,
    R& result);

// sdotc -> sdotu
template <typename ExecutionPolicy,
          typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const float* x, int incX,
     const float* y, int incY,
     R& result);

// ddot -> ddotu
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const double* x, int incX,
    const double* y, int incY,
    R& result);

// ddotc -> ddotu
template <typename ExecutionPolicy,
          typename R>
void
dotc(const ExecutionPolicy& exec,
     int n,
     const double* x, int incX,
     const double* y, int incY,
     R& result);

// cdot -> cdotc
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    R& result);

// zdot -> zdotc
template <typename ExecutionPolicy,
          typename R>
void
dot(const ExecutionPolicy& exec,
    int n,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    R& result);

} // end namespace generic
} // end namespace system
} // end namespace blam

#include <blam/system/generic/detail/dot.inl>
