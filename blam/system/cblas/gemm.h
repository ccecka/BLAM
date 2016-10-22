#pragma once

#include <blam/detail/config.h>
#include <blam/system/cblas/execution_policy.h>

namespace blam
{
namespace cblas
{

// sgemm
void
gemm(const CBLAS_LAYOUT order,
     const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     const float& alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float& beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_sgemm");

  cblas_sgemm(order,
              transA, transB,
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// dgemm
void
gemm(const CBLAS_LAYOUT order,
     const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     const double& alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     const double& beta,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_dgemm");

  cblas_dgemm(order,
              transA, transB,
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// cgemm
void
gemm(const CBLAS_LAYOUT order,
     const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     const ComplexFloat& alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     const ComplexFloat& beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_cgemm");

  cblas_cgemm(order,
              transA, transB,
              m, n, k,
              reinterpret_cast<const float*>(&alpha),
              reinterpret_cast<const float*>(A), ldA,
              reinterpret_cast<const float*>(B), ldB,
              reinterpret_cast<const float*>(&beta),
              reinterpret_cast<float*>(C), ldC);
}

// zgemm
void
gemm(const CBLAS_LAYOUT order,
     const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     const ComplexDouble& alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     const ComplexDouble& beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cblas_zgemm");

  cblas_zgemm(order,
              transA, transB,
              m, n, k,
              reinterpret_cast<const double*>(&alpha),
              reinterpret_cast<const double*>(A), ldA,
              reinterpret_cast<const double*>(B), ldB,
              reinterpret_cast<const double*>(&beta),
              reinterpret_cast<double*>(C), ldC);
}

// blam -> cblas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
auto
gemm(const execution_policy<DerivedPolicy>& /*exec*/,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
    -> decltype(gemm(cblas_order(order),
                     cblas_transpose(transA), cblas_transpose(transB),
                     m, n, k,
                     alpha,
                     A, ldA,
                     B, ldB,
                     beta,
                     C, ldC))
{
  return gemm(cblas_order(order),
              cblas_transpose(transA), cblas_transpose(transB),
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

} // end namespace cblas
} // end namespace blam
