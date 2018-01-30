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

#include <vector>

#include "disable_function.h"
#include "print_type.h"

#define BLAM_USE_DECAY
#include <blam/blas/level3/gemm_batch.h>

#include <blam/system/mkl/mkl.h>


namespace mine {

template <class DerivedPolicy>
struct execution_policy : public DerivedPolicy {
  mutable std::string prefix_;
};


template <class Function, class DerivedPolicy, class... T>
void
invoke(const execution_policy<DerivedPolicy>& exec, Function f, T&&... t)
{
  std::cout << exec.prefix_ << type_name<Function>() << "(" << type_name<DerivedPolicy>() << ", ";
  print_all(std::forward<T>(t)...);
  std::cout << ")" << std::endl;
  exec.prefix_ += "  ";

  using namespace experimental;
  f(remove_customization_point<Function>(exec), std::forward<T>(t)...);

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
  std::vector<T> A(m*k, T(0.5));
  std::vector<T> B(k*n*p, T(2.0));
  std::vector<T> C(m*n*p, T(0.0));

  blam::gemm_batch(exec,
                   m, n, k,
                   alpha,
                   A.data(), m, 0,
                   B.data(), k, k*n,
                   beta,
                   C.data(), m, m*n,
                   p);
}


int main()
{
  {
  mine::execution_policy<blam::mkl::tag> exec;
  test<float>(exec, 4);
  }

  std::cout << "\n################################################\n" << std::endl;

  {
  mine::execution_policy<blam::cblas::tag> exec;
  test<float>(exec, 4);
  }
}
