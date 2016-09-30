#pragma once

#include <blam/detail/config.h>

#include <blam/adl/gemv.h>
//#include <blam/adl/dot.h>

namespace blam
{
namespace system
{
namespace generic
{

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
gemv(const ExecutionPolicy& exec,
     StorageOrder order, Transpose trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
{
#if defined(BLAM_USE_DECAY)
  // dot
#else
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
#endif
}

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
gemv(const ExecutionPolicy& exec,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
{
  blam::adl::gemv(exec, NoTrans,
                  m, n,
                  alpha,
                  A, ldA,
                  x, incX,
                  beta,
                  y, incY);
}

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename VX,
          typename Beta, typename VY>
void
gemv(const ExecutionPolicy& exec,
     Transpose trans,
     int m, int n,
     const Alpha& alpha,
     const MA* A, int ldA,
     const VX* x, int incX,
     const Beta& beta,
     VY* y, int incY)
{
  blam::adl::gemv(exec, ColMajor, NoTrans,
                  m, n,
                  alpha,
                  A, ldA,
                  x, incX,
                  beta,
                  y, incY);
}

} // end namespace generic
} // end namespace system
} // end namespace blam
