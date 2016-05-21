#pragma once

#include <blam/detail/config.h>

#include <blam/adl/gemm.h>
//#include <blam/adl/gemv.h>

namespace blam
{
namespace system
{
namespace generic
{

// Swap to remove dependence on StorageOrder
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
gemm(const ExecutionPolicy& exec,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
{
#if defined(BLAM_USE_DECAY)
  // gemv
#else
  static_assert(sizeof(ExecutionPolicy) == 0, "BLAM UNIMPLEMENTED");
#endif
}

// Default to ColMajor
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
gemm(const ExecutionPolicy& exec,
     Transpose transA, Transpose transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
{
  blam::adl::gemm(exec,
                  ColMajor, transA, transB,
                  m, n, k,
                  alpha,
                  A, ldA,
                  B, ldB,
                  beta,
                  C, ldC);
}

// Default to NoTrans
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
gemm(const ExecutionPolicy& exec,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
{
  blam::adl::gemm(exec,
                  NoTrans, NoTrans,
                  m, n, k,
                  alpha,
                  A, ldA,
                  B, ldB,
                  beta,
                  C, ldC);
}

} // end namespace generic
} // end namespace system
} // end namespace blam
