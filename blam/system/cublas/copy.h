#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// scopy
void
copy(cublasHandle_t handle, int n,
     const float* x, int incX,
     float* y, int incY)
{
  BLAM_DEBUG_OUT("cublasScopy");

  cublasScopy(handle, n,
              x, incX,
              y, incY);
}

// dcopy
void
copy(cublasHandle_t handle, int n,
     const double* x, int incX,
     double* y, int incY)
{
  BLAM_DEBUG_OUT("cublasDcopy");

  cublasDcopy(handle, n,
              x, incX,
              y, incY);
}

// ccopy
void
copy(cublasHandle_t handle, int n,
     const ComplexFloat* x, int incX,
     ComplexFloat* y, int incY)
{
  BLAM_DEBUG_OUT("cublasCcopy");

  cublasCcopy(handle, n,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<cuFloatComplex*>(y), incY);
}

// zcopy
void
copy(cublasHandle_t handle, int n,
     const ComplexDouble* x, int incX,
     ComplexDouble* y, int incY)
{
  BLAM_DEBUG_OUT("cublasZcopy");

  cublasZcopy(handle, n,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<cuDoubleComplex*>(y), incY);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY>
auto
copy(const execution_policy<DerivedPolicy>& exec, int n,
     const VX* x, int incX,
     VY* y, int incY)
    -> decltype(copy(handle(derived_cast(exec)), n,
                     x, incX,
                     y, incY))
{
  return copy(handle(derived_cast(exec)), n,
              x, incX,
              y, incY);
}

} // end namespace cublas
} // end namespace blam
