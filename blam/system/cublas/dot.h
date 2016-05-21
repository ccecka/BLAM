#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sdot
template <typename DerivedPolicy>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const float* x, int incX,
    const float* y, int incY,
    float& result)
{
    BLAM_DEBUG_OUT("cublasSdot");

    cublasSdot(handle(derived_cast(exec)), n,
               x, incX,
               y, incY,
               &result);
}

// ddot
template <typename DerivedPolicy>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const double* x, int incX,
    const double* y, int incY,
    double& result)
{
    BLAM_DEBUG_OUT("cublasDdot");

    cublasDdot(handle(derived_cast(exec)), n,
               x, incX,
               y, incY,
               &result);
}

// cdotc
template <typename DerivedPolicy>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const ComplexFloat* x, int incX,
    const ComplexFloat* y, int incY,
    ComplexFloat& result)
{
    BLAM_DEBUG_OUT("cublasCdotc");

    cublasCdotc(handle(derived_cast(exec)), n,
                reinterpret_cast<const cuFloatComplex*>(x), incX,
                reinterpret_cast<const cuFloatComplex*>(y), incY,
                reinterpret_cast<cuFloatComplex*>(&result));
}

// zdotc
template <typename DerivedPolicy>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const ComplexDouble* x, int incX,
    const ComplexDouble* y, int incY,
    ComplexDouble& result)
{
    BLAM_DEBUG_OUT("cublasZdotc");

    cublasZdotc(handle(derived_cast(exec)), n,
                reinterpret_cast<const cuDoubleComplex*>(x), incX,
                reinterpret_cast<const cuDoubleComplex*>(y), incY,
                reinterpret_cast<cuDoubleComplex*>(&result));
}

// cdotu
template <typename DerivedPolicy>
void
dotu(const execution_policy<DerivedPolicy>& exec,
     int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat& result)
{
    BLAM_DEBUG_OUT("cublasCdotu");

    cublasCdotu(handle(derived_cast(exec)), n,
                reinterpret_cast<const cuFloatComplex*>(x), incX,
                reinterpret_cast<const cuFloatComplex*>(y), incY,
                reinterpret_cast<cuFloatComplex*>(&result));
}

// zdotu
template <typename DerivedPolicy>
void
dotu(const execution_policy<DerivedPolicy>& exec,
     int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble& result)
{
    BLAM_DEBUG_OUT("cublasZdotu");

    cublasZdotu(handle(derived_cast(exec)), n,
                reinterpret_cast<const cuDoubleComplex*>(x), incX,
                reinterpret_cast<const cuDoubleComplex*>(y), incY,
                reinterpret_cast<cuDoubleComplex*>(&result));
}

} // end namespace cublas
} // end namespace blam
