#pragma once

#include <blam/detail/config.h>
#include <blam/adl/batch_gemm.h>

namespace blam
{

#if 0
template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
batch_gemm(ExecutionPolicy&& exec,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int p);

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
batch_gemm(ExecutionPolicy&& exec,
           Transpose transA, Transpose transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int p);

template <typename ExecutionPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
batch_gemm(ExecutionPolicy&& exec,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int p);
#endif

using blam::adl::batch_gemm;

} // end namespace blam
