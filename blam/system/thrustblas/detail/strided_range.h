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

#include <blam/detail/config.h>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace blam
{

/*! \brief RandomAccessIterator for strided access to array entries.
 *
 * \tparam RandomAccessIterator The iterator type used to encapsulate the underlying data.
 *
 * \par Overview
 * \p strided_range represents a strided range entries in a underlying array.
 *  This iterator is useful for creating a strided sublist of entries from a range.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a \p strided_range whose
 *  \c value_type is \c int and whose values are gather from a \p counting_array.
 *
 *  \code
 *  #include <blam/array1d.h>
 *  #include <blam/iterator/strided_range.h>
 *
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    typedef blam::counting_array<int>::iterator Iterator;
 *
 *    blam::counting_array<int> a(30);
 *    blam::strided_range<Iterator> iter(a.begin(), a.end(), 5);
 *
 *    std::cout << iter[0] << std::endl;   // returns 0
 *    std::cout << iter[1] << std::endl;   // returns 5
 *    std::cout << iter[3] << std::endl;   // returns 15
 *
 *    return 0;
 *  }
 *  \endcode
 */
template <typename RandomAccessIterator>
class strided_range
{
 public:

  /*! \cond */
  typedef typename thrust::iterator_value<RandomAccessIterator>::type                       value_type;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type                      memory_space;
  typedef typename thrust::iterator_pointer<RandomAccessIterator>::type                     pointer;
  typedef typename thrust::iterator_reference<RandomAccessIterator>::type                   reference;
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  difference_type;
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  size_type;

  struct StrideFunctor {
    typedef difference_type result_type;
    difference_type s_;
    StrideFunctor(const difference_type& s) : s_(s) {}
    __host__ __device__
    difference_type operator()(const difference_type& n) const {
      return n*s_;
    }
  };
  typedef typename thrust::counting_iterator<difference_type>                               CountingIterator;
  typedef typename thrust::transform_iterator<StrideFunctor, CountingIterator>              TransformIterator;
  typedef typename thrust::permutation_iterator<RandomAccessIterator,TransformIterator>     PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;
  /*! \endcond */

  /*! \brief Null constructor initializes this \p strided_range's stride to zero.
   */
  strided_range(void)
      : stride(0) {}

  /*! \brief This constructor builds a \p strided_range from a range.
   *  \param first The beginning of the range.
   *  \param last The end of the range.
   *  \param stride The stride between consecutive entries in the iterator.
   */
  strided_range(RandomAccessIterator first, RandomAccessIterator last, difference_type stride)
      : first(first), last(last), stride(stride) {}

  /*! \brief This method returns an iterator pointing to the beginning of
   *  this strided sequence of entries.
   *  \return mStart
   */
  iterator begin(void) const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0),
                                                        StrideFunctor(stride)));
  }

  /*! \brief This method returns an iterator pointing to one element past
   *  the last of this strided sequence of entries.
   *  \return mEnd
   */
  iterator end(void) const
  {
    return begin() + (thrust::distance(first,last) + (stride - 1)) / stride;
  }

  /*! \brief Subscript access to the data contained in this iterator.
   *  \param n The index of the element for which data should be accessed.
   *  \return Read/write reference to data.
   *
   *  This operator allows for easy, array-style, data access.
   *  Note that data access with this operator is unchecked and
   *  out_of_range lookups are not defined.
   */
  reference operator[](size_type n) const
  {
    return *(begin() + n);
  }

  static iterator make(RandomAccessIterator first, difference_type stride)
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0),
                                                        StrideFunctor(stride)));
  }

 protected:

  /*! \cond */
  RandomAccessIterator first;
  RandomAccessIterator last;
  difference_type stride;
  /*! \endcond */

}; // end strided_range

template <typename RandomAccessIterator, typename Int>
strided_range<RandomAccessIterator>
make_strided_range(RandomAccessIterator first, RandomAccessIterator last, Int stride) {
  return strided_range<RandomAccessIterator>(first, last, stride);
}

template <typename RandomAccessIterator, typename Int>
typename strided_range<RandomAccessIterator>::iterator
make_strided_iterator(RandomAccessIterator it, Int stride) {
  return strided_range<RandomAccessIterator>::make(it, stride);
}

} // end namespace blam
