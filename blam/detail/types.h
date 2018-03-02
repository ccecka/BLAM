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
#include <type_traits>

namespace blam
{

// User could potentially define ComplexFloat/ComplexDouble instead of std::
#ifndef BLAM_COMPLEX_TYPES
#define BLAM_COMPLEX_TYPES 1
template <typename T>
using Complex       = std::complex<T>;
using ComplexFloat  = Complex<float>;
using ComplexDouble = Complex<double>;
#endif // BLAM_COMPLEX_TYPES

// -----------------------------------------------------------------------------
// Compile-time checks on blam::Complex
static_assert(std::is_same<Complex<float>, ComplexFloat>::value, "Complex<float> != ComplexFloat");
static_assert(std::is_same<Complex<double>, ComplexDouble>::value, "Complex<double> != ComplexDouble");
static_assert(sizeof(ComplexFloat) == 2*sizeof(float), "ComplexFloat should model LiteralType");    // XXX: improve
static_assert(sizeof(ComplexDouble) == 2*sizeof(double), "ComplexDouble should model LiteralType"); // XXX: improve

// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template <typename T>
inline T
abs1(const T& v) {
  using std::abs;
  return abs(v);
}

template <typename T>
inline T
abs1(const Complex<T>& v) {
  using std::real; using std::imag;
  return abs1(real(v)) + abs1(imag(v));
}

//
// -----------------------------------------------------------------------------
//

enum Layout : char {
  ColMajor = 'C',
  RowMajor = 'R'
};

enum Op : char {
  NoTrans   = 'N',
  Conj      = 'X',
  Trans     = 'T',
  ConjTrans = 'C'
};

enum Uplo : char {
  Upper = 'U',
  Lower = 'L',
  General   = 'G'
};

enum Diag : char {
  Unit    = 'U',
  NonUnit = 'N'
};

enum Side : char {
  Left  = 'L',
  Right = 'R'
};

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style char.

inline char
to_char(Layout layout)
{
  return char(layout);
}

inline char
to_char(Op op)
{
  return char(op);
}

inline char
to_char(Uplo uplo)
{
  return char(uplo);
}

inline char
to_char(Diag diag)
{
  return char(diag);
}

inline char
to_char(Side side)
{
  return char(side);
}

// -----------------------------------------------------------------------------
// Convert LAPACK-style char to enum.

inline Layout
char2layout(char layout)
{
  layout = (char) toupper(layout);
  assert(layout == 'C' || layout == 'R');
  return Layout(layout);
}

inline Op
char2op(char op)
{
  op = (char) toupper(op);
  assert(op == 'N' || op == 'T' || op == 'C');
  return Op(op);
}

inline Uplo
char2uplo(char uplo)
{
  uplo = (char) toupper(uplo);
  assert(uplo == 'L' || uplo == 'U' || uplo == 'G');
  return Uplo(uplo);
}

inline Diag
char2diag(char diag)
{
  diag = (char) toupper(diag);
  assert(diag == 'N' || diag == 'U');
  return Diag(diag);
}

inline Side
char2side(char side)
{
  side = (char) toupper(side);
  assert(side == 'L' || side == 'R');
  return Side(side);
}

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style string.

inline const char*
to_string(Layout layout)
{
  switch (layout) {
    case Layout::ColMajor: return "col";
    case Layout::RowMajor: return "row";
  }
  return "<unknown>";
}

inline const char*
to_string(Op op)
{
  switch (op) {
    case Op::NoTrans:   return "notrans";
    case Op::Trans:     return "trans";
    case Op::ConjTrans: return "conj";
    case Op::Conj:      return "";
  }
  return "<unknown>";
}

inline const char*
to_string(Uplo uplo)
{
  switch (uplo) {
    case Uplo::Lower:   return "lower";
    case Uplo::Upper:   return "upper";
    case Uplo::General: return "general";
  }
  return "<unknown>";
}

inline const char*
to_string(Diag diag)
{
  switch (diag) {
    case Diag::NonUnit: return "nonunit";
    case Diag::Unit:    return "unit";
  }
  return "<unknown>";
}

inline const char*
to_string(Side side)
{
  switch (side) {
    case Side::Left:  return "left";
    case Side::Right: return "right";
  }
  return "<unknown>";
}

} // end namespace blam
