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
          typename Alpha, typename T, typename U>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const Alpha& alpha,
    const T* x, int incX,
    const T* y, int incY,
    U* A, int ldA);

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename T, typename U>
void
ger(const ExecutionPolicy& exec,
    int m, int n,
    const Alpha& alpha,
    const T* x, int incX,
    const T* y, int incY,
    U* A, int ldA);

template <typename ExecutionPolicy,
          typename Alpha, typename T, typename U>
void
geru(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const T* x, int incX,
     const T* y, int incY,
     U* A, int ldA);

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename T, typename U>
void
geru(const ExecutionPolicy& exec,
     int m, int n,
     const Alpha& alpha,
     const T* x, int incX,
     const T* y, int incY,
     U* A, int ldA);

// sgeru -> sger
template <typename ExecutionPolicy>
void
geru(const ExecutionPolicy& exec,
     int m, int n,
     const float& alpha,
     const float* x, int incX,
     const float* y, int incY,
     float* A, int ldA);

// dgeru -> dger
template <typename ExecutionPolicy>
void
geru(const ExecutionPolicy& exec,
     int m, int n,
     const double& alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA);

} // end namespace generic
} // end namespace system
} // end namespace blam

#include <blam/system/generic/detail/ger.inl>
