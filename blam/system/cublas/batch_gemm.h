#pragma once

#include <blam/detail/config.h>
#include <blam/system/cublas/execution_policy.h>

#if (CUDA_VERSION >= 8000)

namespace blam
{
namespace cublas
{

// sgemm
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float* beta,
           float* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasSgemmStridedBatched");

  cublasSgemmStridedBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            A, ldA, loA,
                            B, ldB, loB,
                            beta,
                            C, ldC, loC,
                            batch_size);
}

// dgemm
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double* beta,
           double* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasDgemmStridedBatched");

  cublasDgemmStridedBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            A, ldA, loA,
                            B, ldB, loB,
                            beta,
                            C, ldC, loC,
                            batch_size);
}

// cgemm
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const ComplexFloat* alpha,
           const ComplexFloat* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const ComplexFloat* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasCgemmStridedBatched");

  cublasCgemmStridedBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const cuFloatComplex*>(alpha),
                            reinterpret_cast<const cuFloatComplex*>(A), ldA, loA,
                            reinterpret_cast<const cuFloatComplex*>(B), ldB, loB,
                            reinterpret_cast<const cuFloatComplex*>(beta),
                            reinterpret_cast<cuFloatComplex*>(C), ldC, loC,
                            batch_size);
}

// zgemm
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const ComplexDouble* alpha,
           const ComplexDouble* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const ComplexDouble* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasZgemmStridedBatched");

  cublasZgemmStridedBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const cuDoubleComplex*>(alpha),
                            reinterpret_cast<const cuDoubleComplex*>(A), ldA, loA,
                            reinterpret_cast<const cuDoubleComplex*>(B), ldB, loB,
                            reinterpret_cast<const cuDoubleComplex*>(beta),
                            reinterpret_cast<cuDoubleComplex*>(C), ldC, loC,
                            batch_size);
}

// scgemm    XXX: Move to general?
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const ComplexFloat* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasS[C]gemmStridedBatched");

  assert(transA == CUBLAS_OP_N);

  cublasSgemmStridedBatched(handle, transA, transB,
                            2*m, n, k,
                            reinterpret_cast<const float*>(alpha),
                            reinterpret_cast<const float*>(A), 2*ldA, 2*loA,
                            reinterpret_cast<const float*>(B), ldB, loB,
                            reinterpret_cast<const float*>(beta),
                            reinterpret_cast<float*>(C), 2*ldC, 2*loC,
                            batch_size);
}

// csgemm    XXX: Move to general?
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const float* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasS[C]gemmStridedBatched");

  assert(transB == CUBLAS_OP_T);

  cublasSgemmStridedBatched(handle, transA, transB,
                            m, 2*n, k,
                            reinterpret_cast<const float*>(alpha),
                            reinterpret_cast<const float*>(A), ldA, loA,
                            reinterpret_cast<const float*>(B), 2*ldB, 2*loB,
                            reinterpret_cast<const float*>(beta),
                            reinterpret_cast<float*>(C), 2*ldC, 2*loC,
                            batch_size);
}

// zdgemm    XXX: Move to general?
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const ComplexDouble* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasD[Z]gemmStridedBatched");

  assert(transA == CUBLAS_OP_N);

  cublasDgemmStridedBatched(handle, transA, transB,
                            2*m, n, k,
                            reinterpret_cast<const double*>(alpha),
                            reinterpret_cast<const double*>(A), 2*ldA, 2*loA,
                            reinterpret_cast<const double*>(B), ldB, loB,
                            reinterpret_cast<const double*>(beta),
                            reinterpret_cast<double*>(C), 2*ldC, 2*loC,
                            batch_size);
}

// dzgemm    XXX: Move to general?
void
batch_gemm(cublasHandle_t handle,
           cublasOperation_t transA, cublasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const double* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("cublasD[Z]gemmStridedBatched");

  assert(transB == CUBLAS_OP_T);

  cublasDgemmStridedBatched(handle, transA, transB,
                            m, 2*n, k,
                            reinterpret_cast<const double*>(alpha),
                            reinterpret_cast<const double*>(A), ldA, loA,
                            reinterpret_cast<const double*>(B), 2*ldB, 2*loB,
                            reinterpret_cast<const double*>(beta),
                            reinterpret_cast<double*>(C), 2*ldC, 2*loC,
                            batch_size);
}

// blam -> cublas
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
batch_gemm(const execution_policy<DerivedPolicy>& exec,
           Transpose transA, Transpose transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int batch_size)
{
  return batch_gemm(handle(derived_cast(exec)),
                    cublas_transpose(transA), cublas_transpose(transB),
                    m, n, k,
                    &alpha,
                    A, ldA, loA,
                    B, ldB, loB,
                    &beta,
                    C, ldC, loC,
                    batch_size);
}

// RowMajor -> ColMajor
template <typename DerivedPolicy,
          typename Alpha, typename MA, typename MB,
          typename Beta, typename MC>
void
batch_gemm(const execution_policy<DerivedPolicy>& exec,
           StorageOrder order, Transpose transA, Transpose transB,
           int m, int n, int k,
           const Alpha& alpha,
           const MA* A, int ldA, int loA,
           const MB* B, int ldB, int loB,
           const Beta& beta,
           MC* C, int ldC, int loC,
           int batch_size)
{
  if (order == ColMajor) {
    return batch_gemm(exec, transA, transB,
                      m, n, k,
                      alpha,
                      A, ldA, loA,
                      B, ldB, loB,
                      beta,
                      C, ldC, loC,
                      batch_size);
  } else { // RowMajor: swap A & B
    return batch_gemm(exec, transB, transA,
                      n, m, k,
                      alpha,
                      B, ldB, loB,
                      A, ldA, loA,
                      beta,
                      C, ldC, loC,
                      batch_size);
  }
}

} // end namespace cublas
} // end namespace blam

#endif // CUDA_VERSION >= 8000
