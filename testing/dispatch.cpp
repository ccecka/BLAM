#define BLAM_DEBUG 1
#include <blam/dot.h>

#include <blam/system/mkl/mkl.h>
#include <blam/system/cublas/cublas.h>
#include <blam/system/thrustblas/thrustblas.h>

#include <thrust/system/omp/execution_policy.h>

int main()
{
  const int n = 4;
  double xa[n] = {1, 2, 3.14, 5};
  double ya[n] = {2, 3, 1,    1};
  double* x = xa;
  double* y = ya;

  double result = 123;

  blam::dot(blam::mkl::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dotu(blam::mkl::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dot(blam::mkl::par, n, x, y, result);
  std::cout << result << std::endl;

  blam::dot(blam::cblas::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dotu(blam::cblas::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dot(blam::cblas::par, n, x, y, result);
  std::cout << result << std::endl;

  blam::dot(blam::cublas::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dotu(blam::cublas::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dot(blam::cublas::par, n, x, y, result);
  std::cout << result << std::endl;

  blam::dot(thrust::cpp::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dotu(thrust::cpp::par, n, x, 1, y, 1, result);
  std::cout << result << std::endl;

  blam::dot(thrust::cpp::par, n, x, y, result);
  std::cout << result << std::endl;

  blam::dot(thrust::omp::par, n, x, y, result);
  std::cout << result << std::endl;
}
