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

#include "disable_signature.h"
#include "print_type.h"

#define BLAM_USE_DECAY
#include <blam/batch_gemm.h>
#include <blam/system/mkl/mkl.h>

#include <thrust/host_vector.h>

namespace mine {

template <class DerivedPolicy>
struct execution_policy : DerivedPolicy {
  mutable std::string prefix_;
};


template <class Function, class DerivedPolicy, class... T>
void
mutate(Function f, const execution_policy<DerivedPolicy>& exec, T&&... t)
{
  std::cout << exec.prefix_ << type_name<Function>() << "(" << type_name<DerivedPolicy>() << ", ";
  print_all(std::forward<T>(t)...);
  std::cout << ")" << std::endl;
  exec.prefix_ += "  ";

  using namespace experimental;
  f(remove_customization_point<signature<Function, T...>>(exec), std::forward<T>(t)...);

  exec.prefix_.erase(exec.prefix_.size()-2);
}

} // end namespace mine


template <typename T, typename ExecutionPolicy>
void
test(const ExecutionPolicy& exec, int n)
{
  int m = n;
  int k = n;
  int p = n;

  T alpha = 1.0, beta = 0.0;
  thrust::host_vector<T> A(m*k, T(0.5));
  thrust::host_vector<T> B(k*n*p, T(2.0));
  thrust::host_vector<T> C(m*n*p, T(0.0));

  blam::batch_gemm(exec,
                   m, n, k,
                   alpha,
                   thrust::raw_pointer_cast(A.data()), m, 0,
                   thrust::raw_pointer_cast(B.data()), k, k*n,
                   beta,
                   thrust::raw_pointer_cast(C.data()), m, m*n,
                   p);
}


int main()
{
  {
  mine::execution_policy<blam::mkl::tag> exec;
  test<float>(exec, 4);
  }

  {
  mine::execution_policy<blam::cblas::tag> exec;
  test<float>(exec, 4);
  }
}
