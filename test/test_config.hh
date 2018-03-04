#pragma once

#include <blam/blam.h>

#include <blam/system/mkl/mkl.h>
#include <blam/system/cublas/cublas.h>

namespace blas = blam;
//static auto blam_default_policy = blam::cublas::par;
static auto blam_default_policy = blam::mkl::par;

#include "CPU_Clock.h"
#include "GPU_Clock.h"

// HACK quick and dirty for testing library only
namespace blam
{
namespace cblas
{
template <typename T>
inline T*
create_device_copy(tag, T* x, size_t n)
{
  return x;
}

template <typename T>
inline void
copy_from_device(tag, T* dx, T* x, size_t n)
{
  assert(dx == x);
  return;
}

template <typename T>
inline void
destroy(tag, T* dx)
{
  return;
}

inline CPU_Clock
get_timer(tag)
{
  return {};
}
} // end namespace cblas

namespace cublas
{
template <typename T>
inline T*
create_device_copy(tag, T* x, size_t n)
{
  T* ptr;
  {
  auto status = cudaMalloc((void**)&ptr, n*sizeof(T));
  assert(status == cudaSuccess);
  }
  {
  auto status = cublasSetVector(n, sizeof(T), x, 1, ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);
  }
  return ptr;
}

template <typename T>
inline void
copy_from_device(tag, T* dx, T* x, size_t n)
{
  auto status = cublasGetVector(n, sizeof(T), dx, 1, x, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);
  return;
}

template <typename T>
inline void
destroy(tag, T* dx)
{
  auto status = cudaFree(dx);
  assert(status == cudaSuccess);
}

inline GPU_Clock
get_timer(tag)
{
  return {};
}
} // end namespace cublas
} // end namespace blam
