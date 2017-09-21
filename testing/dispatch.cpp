/******************************************************************************
 * Copyright (C) 2016-2017, Cris Cecka.  All rights reserved.
 * Copyright (C) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

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

  // XXX
  //blam::dotc(blam::cublas::par, n, reinterpret_cast<std::complex<double>*>(x), 1, y, 1, result);
  //std::cout << result << std::endl;

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
