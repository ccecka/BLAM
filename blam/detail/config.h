#pragma once

#include <cassert>

//-- BLAM_DEBUG_OUT ---------------------------------------------------------
#ifdef BLAM_DEBUG
# include <iostream>
# ifndef BLAM_DEBUG_OUT
#  define BLAM_DEBUG_OUT(msg)    std::cerr << "BLAM: " << msg << std::endl
#  define BLAM_DEBUG_OUT_2(msg)  std::cerr << msg << std::endl
# endif // BLAM_DEBUG_OUT
#else
# ifndef BLAM_DEBUG_OUT
#  define BLAM_DEBUG_OUT(msg)
#  define BLAM_DEBUG_OUT_2(msg)
# endif // BLAM_DEBUG_OUT
#endif // BLAM_DEBUG

// XXX: Move to typedef.h?

namespace blam
{

enum StorageOrder {
  ColMajor = 0,
  RowMajor = 1
};

enum Transpose {
  NoTrans   = 0,
  Conj      = 1,
  Trans     = 2,
  ConjTrans = 3
};

} // end namespace blam


// XXX: Move to complex.h?

#include <complex>

namespace blam
{

// User could potentially define ComplexFloat/ComplexDouble instead of thrust::
#ifndef BLAM_COMPLEX_TYPES
#define BLAM_COMPLEX_TYPES 1
template <typename T>
using complex       = std::complex<T>;
using ComplexFloat  = complex<float>;
using ComplexDouble = complex<double>;
#endif // BLAM_COMPLEX_TYPES

} // end namespace blam
