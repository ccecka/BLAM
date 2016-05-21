#pragma once

#include <blam/detail/config.h>

namespace blam
{

__host__ __device__
float conj(const float& v) {
  return v;
}

__host__ __device__
double conj(const double& v) {
  return v;
}

template <typename T>
struct conj_fn
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  typedef T argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result.
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>conj(x)</tt>.
   */
  __host__ __device__
  T operator()(const T& x) const { return conj(x); }
};

}
