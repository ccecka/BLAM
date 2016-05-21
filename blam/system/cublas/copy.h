#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// scopy
template <typename DerivedPolicy>
void
copy(const execution_policy<DerivedPolicy>& exec,
     int n,
     const float* x, int incX,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasScopy");

  cublasScopy(handle(derived_cast(exec)), n,
              x, incX,
              y, incY);
}

// dcopy
template <typename DerivedPolicy>
void
copy(const execution_policy<DerivedPolicy>& exec,
     int n,
     const double* x, int incX,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDcopy");

  cublasDcopy(handle(derived_cast(exec)), n,
              x, incX,
              y, incY);
}

// ccopy
template <typename DerivedPolicy>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const ComplexFloat* x, int incX,
    ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCcopy");

  cublasCcopy(handle(derived_cast(exec)), n,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<cuFloatComplex*>(y), incY);
}

// zcopy
template <typename DerivedPolicy>
void
dot(const execution_policy<DerivedPolicy>& exec,
    int n,
    const ComplexDouble* x, int incX,
    ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZcopy");

  cublasZcopy(handle(derived_cast(exec)), n,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<cuDoubleComplex*>(y), incY);
}

} // end namespace cublas
} // end namespace blam
