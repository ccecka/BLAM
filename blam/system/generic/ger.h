#pragma once

#include <blam/detail/config.h>

namespace blam
{
namespace system
{
namespace generic
{

template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
gerc(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA);

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
gerc(const ExecutionPolicy& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA);

template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
geru(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA);

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
geru(const ExecutionPolicy& exec,
     int m, int n,
     const Alpha& alpha,
     const VX* x, int incX,
     const VY* y, int incY,
     MA* A, int ldA);

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename VX, typename VY, typename MA>
void
ger(const ExecutionPolicy& exec,
    int m, int n,
    const Alpha& alpha,
    const VX* x, int incX,
    const VY* y, int incY,
    MA* A, int ldA);

// sger -> sgeru
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const float& alpha,
    const float* x, int incX,
    const float* y, int incY,
    MA* A, int ldA);

// sgerc -> sgeru
template <typename ExecutionPolicy,
          typename MA>
void
gerc(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const float& alpha,
     const float* x, int incX,
     const float* y, int incY,
     MA* A, int ldA);

// dger -> dgeru
template <typename ExecutionPolicy>
void
geru(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const double& alpha,
     const double* x, int incX,
     const double* y, int incY,
     double* A, int ldA);

// dger -> dgeru
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const double& alpha,
    const double* x, int incX,
    const double* y, int incY,
    MA* A, int ldA);

// dgerc -> dgeru
template <typename ExecutionPolicy,
          typename MA>
void
gerc(const ExecutionPolicy& exec,
     StorageOrder order, int m, int n,
     const double& alpha,
     const double* x, int incX,
     const double* y, int incY,
     MA* A, int ldA);

// cger -> cgerc
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const ComplexFloat& alpha,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    MA* A, int ldA);

// zger -> zgerc
template <typename ExecutionPolicy,
          typename MA>
void
ger(const ExecutionPolicy& exec,
    StorageOrder order, int m, int n,
    const ComplexDouble& alpha,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    MA* A, int ldA);

} // end namespace generic
} // end namespace system
} // end namespace blam

#include <blam/system/generic/detail/ger.inl>
