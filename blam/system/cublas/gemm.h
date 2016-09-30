#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

namespace blam
{
namespace cublas
{

// sgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float* beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasSgemm");

  cublasSgemm(handle, transA, transB,
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// dgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const double* alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     const double* beta,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgemm");

  cublasDgemm(handle, transA, transB,
              m, n, k,
              alpha,
              A, ldA,
              B, ldB,
              beta,
              C, ldC);
}

// cgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     const ComplexFloat* beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasCgemm");

  cublasCgemm(handle, transA, transB,
              m, n, k,
              reinterpret_cast<const cuFloatComplex*>(alpha),
              reinterpret_cast<const cuFloatComplex*>(A), ldA,
              reinterpret_cast<const cuFloatComplex*>(B), ldB,
              reinterpret_cast<const cuFloatComplex*>(beta),
              reinterpret_cast<cuFloatComplex*>(C), ldC);
}

// zgemm
void
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     const ComplexDouble* beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("cublasDgemm");

  cublasZgemm(handle, transA, transB,
              m, n, k,
              reinterpret_cast<const cuDoubleComplex*>(alpha),
              reinterpret_cast<const cuDoubleComplex*>(A), ldA,
              reinterpret_cast<const cuDoubleComplex*>(B), ldB,
              reinterpret_cast<const cuDoubleComplex*>(beta),
              reinterpret_cast<cuDoubleComplex*>(C), ldC);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
gemm(const execution_policy<DerivedPolicy>& exec,
     Transpose transA, Transpose transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
{
  gemm(handle(derived_cast(exec)),
       cublas_transpose(transA), cublas_transpose(transB),
       m, n, k,
       &alpha,
       A, ldA,
       B, ldB,
       &beta,
       C, ldC);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
gemm(const execution_policy<DerivedPolicy>& exec,
     StorageOrder order, Transpose transA, Transpose transB,
     int m, int n, int k,
     const Alpha& alpha,
     const MA* A, int ldA,
     const MB* B, int ldB,
     const Beta& beta,
     MC* C, int ldC)
{
  if (order == ColMajor) {
    gemm(exec, transA, transB,
         m, n, k,
         alpha,
         A, ldA,
         B, ldB,
         beta,
         C, ldC);
  } else { // RowMajor: swap A & B
    gemm(exec, transB, transA,
         n, m, k,
         alpha,
         B, ldB,
         A, ldA,
         beta,
         C, ldC);
  }
}

} // end namespace cublas
} // end namespace blam
