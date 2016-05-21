#pragma once

#include <blam/detail/config.h>
#include <blam/adl/gemv.h>

namespace blam
{

#if 0
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
gemv(ExecutionPolicy&& exec,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY);

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
gemv(ExecutionPolicy&& exec,
     Transpose trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY);

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
gemv(ExecutionPolicy&& exec,
     StorageOrder order, Transpose trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY);

#endif

using blam::adl::gemv;

} // end namespace blam
