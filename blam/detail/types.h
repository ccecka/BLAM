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

#pragma once

#include <complex>

namespace blam
{

// User could potentially define ComplexFloat/ComplexDouble instead of std::
#ifndef BLAM_COMPLEX_TYPES
#define BLAM_COMPLEX_TYPES 1
template <typename T>
using complex       = std::complex<T>;
using ComplexFloat  = complex<float>;
using ComplexDouble = complex<double>;
#endif // BLAM_COMPLEX_TYPES

// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template <typename T>
T abs1(const T& v) {
  using std::abs;
  return abs(v);
}

template <typename T>
T abs1(const complex<T>& v) {
  using std::real; using std::imag;
  return abs1(real(v)) + abs1(imag(v));
}

} // end namespace blam


namespace blam
{

enum Layout : char {
  ColMajor = 0,
  RowMajor = 1
};

enum Op : char {
  NoTrans   = 0,
  Conj      = 1,
  Trans     = 2,
  ConjTrans = 3
};

enum Uplo : char {
  Upper = 'U',
  Lower = 'L'
};

enum Diag : char {
  Unit    = 'U',
  NonUnit = 'N'
};

enum Side : char {
  Left  = 'L',
  Right = 'R'
};

} // end namespace blam
