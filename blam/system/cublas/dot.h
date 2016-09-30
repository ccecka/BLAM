#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sdot
void
dotu(cublasHandle_t handle, int n,
     const float* x, int incX,
     const float* y, int incY,
     float* result)
{
  BLAM_DEBUG_OUT("cublasSdot");

  cublasSdot(handle, n,
             x, incX,
             y, incY,
             result);
}

// ddot
void
dotu(cublasHandle_t handle, int n,
     const double* x, int incX,
     const double* y, int incY,
     double* result)
{
  BLAM_DEBUG_OUT("cublasDdot");

  cublasDdot(handle, n,
             x, incX,
             y, incY,
             result);
}

// sdot
void
dotc(cublasHandle_t handle, int n,
     const float* x, int incX,
     const float* y, int incY,
     float* result)
{
  BLAM_DEBUG_OUT("cublasSdot");

  cublasSdot(handle, n,
             x, incX,
             y, incY,
             result);
}

// ddot
void
dotc(cublasHandle_t handle, int n,
     const double* x, int incX,
     const double* y, int incY,
     double* result)
{
  BLAM_DEBUG_OUT("cublasDdot");

  cublasDdot(handle, n,
             x, incX,
             y, incY,
             result);
}

// cdotc
void
dotc(cublasHandle_t handle, int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* result)
{
  BLAM_DEBUG_OUT("cublasCdotc");

  cublasCdotc(handle, n,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(y), incY,
              reinterpret_cast<cuFloatComplex*>(result));
}

// zdotc
void
dotc(cublasHandle_t handle, int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* result)
{
  BLAM_DEBUG_OUT("cublasZdotc");

  cublasZdotc(handle, n,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(y), incY,
              reinterpret_cast<cuDoubleComplex*>(result));
}

// cdotu
void
dotu(cublasHandle_t handle, int n,
     const ComplexFloat* x, int incX,
     const ComplexFloat* y, int incY,
     ComplexFloat* result)
{
  BLAM_DEBUG_OUT("cublasCdotu");

  cublasCdotu(handle, n,
              reinterpret_cast<const cuFloatComplex*>(x), incX,
              reinterpret_cast<const cuFloatComplex*>(y), incY,
              reinterpret_cast<cuFloatComplex*>(result));
}

// zdotu
void
dotu(cublasHandle_t handle, int n,
     const ComplexDouble* x, int incX,
     const ComplexDouble* y, int incY,
     ComplexDouble* result)
{
  BLAM_DEBUG_OUT("cublasZdotu");

  cublasZdotu(handle, n,
              reinterpret_cast<const cuDoubleComplex*>(x), incX,
              reinterpret_cast<const cuDoubleComplex*>(y), incY,
              reinterpret_cast<cuDoubleComplex*>(result));
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
void
dotc(const execution_policy<DerivedPolicy>& exec,
     int n,
     const VX* x, int incX,
     const VY* y, int incY,
     R& result)
{
  dotc(handle(derived_cast(exec)), n,
       x, incX,
       y, incY,
       &result);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename VX, typename VY, typename R>
void
dotu(const execution_policy<DerivedPolicy>& exec,
    int n,
    const VX* x, int incX,
    const VY* y, int incY,
    R& result)
{
  dotu(handle(derived_cast(exec)), n,
       x, incX,
       y, incY,
       &result);
}

} // end namespace cublas
} // end namespace blam
